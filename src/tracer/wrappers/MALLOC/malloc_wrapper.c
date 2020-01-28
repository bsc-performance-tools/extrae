/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

#include "common.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef __USE_GNU
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_ASSERT_H
# include <assert.h>
#endif
#ifdef HAVE_MEMKIND_H
# include <memkind.h>
#endif

#include "wrapper.h"
#include "trace_macros.h"
#include "malloc_probe.h"

// #define DEBUG

#if defined(INSTRUMENT_DYNAMIC_MEMORY)

/*
	This wrappers can only be compiled if the file is being compiled to
	generate a shared library (-DPIC)
*/

static void* (*real_malloc)(size_t) = NULL;
static void (*real_free)(void *) = NULL;
static void* (*real_calloc)(size_t, size_t) = NULL;
static void* (*real_realloc)(void*, size_t) = NULL;
static int   (*real_posix_memalign)(void **, size_t, size_t) = NULL;

# if defined(HAVE_MEMKIND)
static void* (*real_memkind_malloc)(memkind_t, size_t) = NULL;
static void* (*real_memkind_calloc)(memkind_t, size_t, size_t) = NULL;
static void* (*real_memkind_realloc)(memkind_t, void *, size_t) = NULL;
static int   (*real_memkind_posix_memalign)(memkind_t, void **, size_t, size_t) = NULL;
static void  (*real_memkind_free)(memkind_t, void *) = NULL; 
# endif

# if defined(HAVE_OPENMP)
static void* (*real_kmpc_malloc)(size_t) = NULL;
static void* (*real_kmpc_aligned_malloc)(size_t, size_t) = NULL;
static void* (*real_kmpc_calloc)(size_t, size_t) = NULL;
static void* (*real_kmpc_realloc)(void *, size_t) = NULL;
static void (*real_kmpc_free)(void *) = NULL;
# endif

/* Note on the implementation!
   We will only instrument those malloc(), realloc() that are larger than
   a given threshold. Therefore, we will only instrument the free() associated to
   those allocations. To this end, we store in malloc entries a vector of pointers
   returned by malloc/realloc that surpass the threshold and that may need later
   instrumentation of their respective free. */
static void ** mallocentries = NULL;
/* Capture also the size of the allocations */
static size_t * mallocentries_sz = NULL;
static unsigned nmallocentries_allocated = 0;
static unsigned nmallocentries = 0;
#define NMALLOCENTRIES_MALLOC 16*1024

/* Registers a new address to be tracked for future free() calls */
static pthread_mutex_t mutex_allocations = PTHREAD_MUTEX_INITIALIZER;
static void Extrae_malloctrace_add (void *p, size_t s)
{
	if (p != NULL)
	{
		unsigned u;
		assert (real_realloc != NULL);

		pthread_mutex_lock (&mutex_allocations);
	
		if (nmallocentries == nmallocentries_allocated)
		{
			mallocentries = real_realloc (mallocentries,
			  (nmallocentries_allocated+NMALLOCENTRIES_MALLOC) * sizeof(void*));
			assert (mallocentries != NULL);
			mallocentries_sz = (size_t*) real_realloc (mallocentries_sz,
			  (nmallocentries_allocated+NMALLOCENTRIES_MALLOC) * sizeof(size_t));
			assert (mallocentries != NULL);
			for (u = nmallocentries_allocated;
			     u < nmallocentries_allocated+NMALLOCENTRIES_MALLOC;
			     u++)
				mallocentries[u] = NULL;
			nmallocentries_allocated += NMALLOCENTRIES_MALLOC;
		}
	
		for (u = 0; u < nmallocentries_allocated; u++)
			if (mallocentries[u] == NULL)
			{
				mallocentries[u]    = p;
				mallocentries_sz[u] = s;
				nmallocentries++;
				break;
			}

		pthread_mutex_unlock (&mutex_allocations);
	}
}

/* Removes an entry from the list of registered addresses */
static int Extrae_malloctrace_remove (const void *p)
{
	unsigned found = FALSE;
	if (p != NULL)
	{
		unsigned u;

		pthread_mutex_lock (&mutex_allocations);

		for (u = 0; u < nmallocentries_allocated; u++)
			if (mallocentries[u] == p)
			{
				mallocentries[u]    = NULL;
				mallocentries_sz[u] = 0;
				nmallocentries--;
				found = TRUE;
				break;
			}

		pthread_mutex_unlock (&mutex_allocations);
	}
	return found;
}

static size_t Extrae_malloctrace_replace (const void *p1, void *p2, size_t s)
{
	size_t prev_sz = 0;

	pthread_mutex_lock (&mutex_allocations);

	int replaced = FALSE;
	if (p1)
	{
		unsigned u;
		for (u = 0; u < nmallocentries_allocated; u++)
			if (mallocentries[u] == p1)
			{
				prev_sz = mallocentries_sz[u];
				mallocentries[u]    = p2;
				mallocentries_sz[u] = s;
				replaced = TRUE;
				break;
			}
	}

	// If we didn't find the pointer, we probably omitted its creation (because of
	// threshold, e.g.). Need to create an entry for this new allocation if it is
	// above the threshold.
	if (!replaced)
	{
		unsigned u;

		if (nmallocentries == nmallocentries_allocated)
		{
			mallocentries = real_realloc (mallocentries,
			  (nmallocentries_allocated+NMALLOCENTRIES_MALLOC) * sizeof(void*));
			assert (mallocentries != NULL);
			mallocentries_sz = (size_t*) real_realloc (mallocentries_sz,
			  (nmallocentries_allocated+NMALLOCENTRIES_MALLOC) * sizeof(size_t));
			assert (mallocentries != NULL);
			for (u = nmallocentries_allocated;
			     u < nmallocentries_allocated+NMALLOCENTRIES_MALLOC;
			     u++)
				mallocentries[u] = NULL;
			nmallocentries_allocated += NMALLOCENTRIES_MALLOC;
		}
	
		for (u = 0; u < nmallocentries_allocated; u++)
			if (mallocentries[u] == NULL)
			{
				mallocentries[u]    = p2;
				mallocentries_sz[u] = s;
				nmallocentries++;
				break;
			}
	}

	pthread_mutex_unlock (&mutex_allocations);

	return prev_sz;
}


/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/

#define TRACE_DYNAMIC_MEMORY_CALLER_IS_ENABLED \
 (Trace_Caller_Enabled[CALLER_DYNAMIC_MEMORY])

#define TRACE_DYNAMIC_MEMORY_CALLER(evttime,offset) \
{ \
	if (TRACE_DYNAMIC_MEMORY_CALLER_IS_ENABLED) \
		Extrae_trace_callers (evttime, offset, CALLER_DYNAMIC_MEMORY); \
}

# if defined(PIC) /* This is only available for .so libraries */
void *malloc (size_t s)
{
	void *res;
	int canInstrument = EXTRAE_INITIALIZED()                 &&
                            mpitrace_on                          &&
                            Extrae_get_trace_malloc()            &&
                            Extrae_get_trace_malloc_allocate()   &&
                            s >= Extrae_get_trace_malloc_allocate_threshold();
	/* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_malloc == NULL)
		real_malloc = EXTRAE_DL_INIT (__func__);

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": malloc is at %p\n", real_malloc);
		fprintf (stderr, PACKAGE_NAME": malloc params %lu\n", s);
	}
#endif

	if (real_malloc != NULL && canInstrument)
	{
		/* If we can instrument, simply capture everything we need 
		   and add the pointer to the list of recorded pointers */
		Backend_Enter_Instrumentation ();
		Probe_Malloc_Entry (s);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_malloc (s);
		if (res != NULL)
		{
			Extrae_malloctrace_add (res, s);
		}
		Probe_Malloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_malloc != NULL)
	{
		/* Otherwise, call the original */
		res = real_malloc (s);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": malloc is not hooked! exiting!!\n");
		abort();
	}

    return res;
}

#define DLSYM_CALLOC_SIZE 8 MB
/*
 * Static buffer to return when calloc is called from within dlsym and we don't
 * have the pointer to the real calloc function
 */
static unsigned char extrae_dlsym_static_buffer[DLSYM_CALLOC_SIZE];

int __in_free = FALSE;

void free (void *p)
{
	if (p == extrae_dlsym_static_buffer) return;

	int canInstrument = EXTRAE_INITIALIZED()                 &&
	                    mpitrace_on                          &&
	                    Extrae_get_trace_malloc();
	/*
	 * Can't be evaluated before, the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized
	 */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	/*
	 * If we don't have the pointer to the real free funtion and we are not
	 * already inside a dlsym, call dlsym
	*/
	if (real_free == NULL && !__in_free)
	{
		__in_free = TRUE;
		real_free = EXTRAE_DL_INIT (__func__);
		__in_free = FALSE;
	}

#if defined(DEBUG)
	if (canInstrument && !__in_free) // fprintf() seems to call free()!
	{
		__in_free = TRUE;
		fprintf(stderr, PACKAGE_NAME": free is at %p\n", real_free);
		fprintf(stderr, PACKAGE_NAME": free params %p\n", p);
		__in_free = FALSE;
	}
#endif

	int present = Extrae_malloctrace_remove (p);

	if (Extrae_get_trace_malloc_free() && real_free != NULL &&
	    canInstrument && present)
	{
		/* If we can instrument, simply capture everything we need and
		   remove the pointer from the list */
		Backend_Enter_Instrumentation();
		Probe_Free_Entry(p);
		real_free(p);
		Probe_Free_Exit();
		Backend_Leave_Instrumentation();
	} else if (real_free != NULL)
	{
		/* Otherwise, call the original */
		real_free(p);
	} else
	{
		/*
		 * If we don't have the real pointer and reach this point, do
		 * nothing and leave the memory unfree'd. This should only
		 * happen once during the initialization.
		*/
	}
}

/* Unfortunately, calloc seems to be invoked if dlsym fails and generates an
infinite loop of recursive calls to calloc */

/* Used to know the depth of calloc calls */
int __in_calloc_depth = 0;
void *calloc (size_t nmemb, size_t size)
{
	__in_calloc_depth++;
	void *res;
	int canInstrument = EXTRAE_INITIALIZED()                 &&
                            mpitrace_on                          &&
                            Extrae_get_trace_malloc();
	/*
	 * Can't be evaluated before because the compiler optimizes the if's
	 * clauses, and THREADID calls a null callback if Extrae is not yet
	 * initialized
	 */
        if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_calloc == NULL)
	{
		if (__in_calloc_depth == 1)
		{
			real_calloc = EXTRAE_DL_INIT (__func__);
		} else if (__in_calloc_depth == 2)
		{
			int i = 0;

			/* Check if the requested size fits in the static buffer */
			if ((nmemb*size) > DLSYM_CALLOC_SIZE)
			{
				fprintf (stderr, PACKAGE_NAME
				    ": The size requested by calloc (%zu) is bigger"
				    " than DLSYM_CALLOC_SIZE, please increase its value and"
				    " recompile.\n", nmemb*size);
				abort();
			}

			/* Zero static buffer before returning it */
			for (i = 0; i<DLSYM_CALLOC_SIZE; i++)
			{
				extrae_dlsym_static_buffer[i] = 0;
			}
			__in_calloc_depth--;
			return extrae_dlsym_static_buffer;
		} else
		{
			/* in_calloc_depth shouldn't be greater than 2 */
			fprintf (stderr, PACKAGE_NAME
			    ": Please turn off calloc instrumentation.\n");
			abort();

		}
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": calloc is at %p\n", real_calloc);
	fprintf (stderr, PACKAGE_NAME": calloc params %u %u\n", nmemb, size);
#endif

	if (real_calloc != NULL && canInstrument)
	{
		Backend_Enter_Instrumentation ();
		Probe_Calloc_Entry (nmemb, size);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_calloc (nmemb, size);
		if (res != NULL)
		{
			Extrae_malloctrace_add (res, size);
		}
		Probe_Calloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_calloc != NULL && !canInstrument)
	{
		/* Otherwise, call the original */
		res = real_calloc (nmemb, size);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": calloc is not hooked! exiting!!\n");
		abort();
	}

	__in_calloc_depth--;
	return res;
}

void *realloc (void *p, size_t s)
{
	void *res;
	int canInstrument = EXTRAE_INITIALIZED()                 &&
                            mpitrace_on                          &&
                            Extrae_get_trace_malloc()            &&
                            Extrae_get_trace_malloc_allocate()   &&
                            s >= Extrae_get_trace_malloc_allocate_threshold();
        /* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
        if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_realloc == NULL)
		real_realloc = EXTRAE_DL_INIT (__func__);

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": realloc is at %p\n", real_realloc);
		fprintf (stderr, PACKAGE_NAME": realloc params %p %lu\n", p, s);
	}
#endif

	if (real_realloc != NULL && canInstrument)
	{
		/* If we can instrument, simply capture everything we need 
		   and remove and add the pointers to the list of recorded pointers */

		int usable_size;

		Backend_Enter_Instrumentation ();
		usable_size = Probe_Realloc_Entry (p, s);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_realloc (p, s);
		if (res != NULL)
		{
			Extrae_malloctrace_replace (p, res, s);
		}
		Probe_Realloc_Exit (res, usable_size);
		Backend_Leave_Instrumentation ();
	}
	else if (real_realloc != NULL)
	{
		/* Otherwise, call the original */
		res = real_realloc (p, s);
		// We may need to remove the previous pointer
		Extrae_malloctrace_remove (p);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": realloc is not hooked! exiting!!\n");
		abort();
	}

    return res;
}

int posix_memalign(void **memptr, size_t alignment, size_t size)
{
  int res = 0;
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_malloc()            &&
                      Extrae_get_trace_malloc_allocate()   &&
                      size >= Extrae_get_trace_malloc_allocate_threshold();
  /* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (real_posix_memalign == NULL)
  {
	real_posix_memalign = EXTRAE_DL_INIT (__func__);
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf(stderr, PACKAGE_NAME": posix_memalign is at %p\n", real_posix_memalign);
    fprintf(stderr, PACKAGE_NAME": posix_memalign params %p %lu %lu\n", memptr, alignment, size);
  }
#endif

  if (real_posix_memalign != NULL && canInstrument)
  {
    Backend_Enter_Instrumentation ();
    Probe_posix_memalign_Entry (size);
    TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
    res = real_posix_memalign(memptr, alignment, size);
    if (res == 0)
    {
      Extrae_malloctrace_add (*memptr, size);
    }
    Probe_memkind_malloc_Exit (*memptr);
    Backend_Leave_Instrumentation ();
  }
  else if (real_posix_memalign != NULL)
  {
    res = real_posix_memalign(memptr, alignment, size);
  }
  else
  {
    fprintf (stderr, PACKAGE_NAME": posix_memalign is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

#  if defined(HAVE_MEMKIND)

static int get_memkind_partition(memkind_t kind)
{
	if (kind == MEMKIND_DEFAULT)
		return MEMKIND_PARTITION_DEFAULT_VAL;
	else if (kind == MEMKIND_HBW)
		return MEMKIND_PARTITION_HBW_VAL;
	else if (kind == MEMKIND_HBW_HUGETLB)
		return MEMKIND_PARTITION_HBW_HUGETLB_VAL;
	else if (kind == MEMKIND_HBW_PREFERRED)
		return MEMKIND_PARTITION_HBW_PREFERRED_VAL;
	else if (kind == MEMKIND_HBW_PREFERRED_HUGETLB)
		return MEMKIND_PARTITION_HBW_PREFERRED_HUGETLB_VAL;
	else if (kind == MEMKIND_HUGETLB)
		return MEMKIND_PARTITION_HUGETLB_VAL;
	else if (kind == MEMKIND_HBW_GBTLB)
		return MEMKIND_PARTITION_HBW_GBTLB_VAL;
	else if (kind == MEMKIND_HBW_PREFERRED_GBTLB)
		return MEMKIND_PARTITION_HBW_PREFERRED_GBTLB_VAL;
	else if (kind == MEMKIND_GBTLB)
		return MEMKIND_PARTITION_GBTLB_VAL;
	else if (kind == MEMKIND_HBW_INTERLEAVE)
		return MEMKIND_PARTITION_HBW_INTERLEAVE_VAL;
	else if (kind == MEMKIND_INTERLEAVE)
		return MEMKIND_PARTITION_INTERLEAVE_VAL;

	return MEMKIND_PARTITION_OTHER_VAL;
}

void *memkind_malloc(memkind_t kind, size_t size)
{
  void *res = NULL;
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_malloc()            &&
                      Extrae_get_trace_malloc_allocate()   &&
                      size >= Extrae_get_trace_malloc_allocate_threshold();
  /* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (real_memkind_malloc == NULL)
  {
	real_memkind_malloc = EXTRAE_DL_INIT (__func__);
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf(stderr, PACKAGE_NAME": memkind_malloc is at %p\n", real_memkind_malloc);
    fprintf(stderr, PACKAGE_NAME": memkind_malloc params %p %lu\n", kind, size);
  } 
#endif

  if (real_memkind_malloc != NULL && canInstrument)
  {
    Backend_Enter_Instrumentation ();
    Probe_memkind_malloc_Entry (get_memkind_partition( kind ), size);
    TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
    res = real_memkind_malloc(kind, size);
    if (res != NULL)
    {
      Extrae_malloctrace_add (res, size);
    }
    Probe_memkind_malloc_Exit (res);
    Backend_Leave_Instrumentation ();
  }
  else if (real_memkind_malloc != NULL)
  {
    res = real_memkind_malloc(kind, size);
  }
  else
  {
    fprintf (stderr, PACKAGE_NAME": memkind_malloc is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

void *memkind_calloc(memkind_t kind, size_t num, size_t size)
{
  void *res = NULL;
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_malloc()            &&
                      Extrae_get_trace_malloc_allocate()   &&
                      size >= Extrae_get_trace_malloc_allocate_threshold();
  /* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (real_memkind_calloc == NULL)
  {
	real_memkind_calloc = EXTRAE_DL_INIT (__func__);
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf(stderr, PACKAGE_NAME": memkind_calloc is at %p\n", real_memkind_calloc);
    fprintf(stderr, PACKAGE_NAME": memkind_calloc params %p %lu %lu\n", kind, num, size);
  }
#endif

  if (real_memkind_calloc != NULL && canInstrument)
  {
    Backend_Enter_Instrumentation ();
    Probe_memkind_calloc_Entry (get_memkind_partition( kind ), num, size);
    TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
    res = real_memkind_calloc(kind, num, size);
    if (res != NULL)
    {
      Extrae_malloctrace_add (res, num*size); 
    }
    Probe_memkind_calloc_Exit (res);
    Backend_Leave_Instrumentation ();
  }
  else if (real_memkind_calloc != NULL)
  {
    res = real_memkind_calloc(kind, num, size);
  }
  else
  {
    fprintf (stderr, PACKAGE_NAME": memkind_calloc is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

void *memkind_realloc(memkind_t kind, void *ptr, size_t size)
{
  void *res = NULL;
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_malloc()            &&
                      Extrae_get_trace_malloc_allocate()   &&
                      size >= Extrae_get_trace_malloc_allocate_threshold();
  /* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (real_memkind_realloc == NULL)
  {
	real_memkind_realloc = EXTRAE_DL_INIT (__func__);
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf(stderr, PACKAGE_NAME": memkind_realloc is at %p\n", real_memkind_realloc);
    fprintf(stderr, PACKAGE_NAME": memkind_realloc params %p %p %lu\n", kind, ptr, size);
  }
#endif

  if (real_memkind_realloc != NULL && canInstrument)
  {

  	int usable_size;

    Backend_Enter_Instrumentation ();
    usable_size = Probe_memkind_realloc_Entry (get_memkind_partition( kind ), ptr, size);
    TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
    res = real_memkind_realloc(kind, ptr, size);
    if (res != NULL)
    {
	  Extrae_malloctrace_replace (ptr, res, size);
    }
    Probe_memkind_realloc_Exit (res, usable_size);
    Backend_Leave_Instrumentation ();
  }
  else if (real_memkind_realloc != NULL)
  {
    res = real_memkind_realloc(kind, ptr, size);
    // We may need to remove the previous pointer
    Extrae_malloctrace_remove (ptr);
  }
  else
  {
    fprintf (stderr, PACKAGE_NAME": memkind_realloc is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

int memkind_posix_memalign(memkind_t kind, void **memptr, size_t alignment, size_t size)
{
  int res = 0;
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_malloc()            &&
                      Extrae_get_trace_malloc_allocate()   &&
                      size >= Extrae_get_trace_malloc_allocate_threshold();
  /* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (real_memkind_posix_memalign == NULL)
  {
	real_memkind_posix_memalign = EXTRAE_DL_INIT (__func__);
  }

#if defined(DEBUG)
  if (canInstrument)
  {
    fprintf(stderr, PACKAGE_NAME": memkind_posix_memalign is at %p\n", real_memkind_posix_memalign);
    fprintf(stderr, PACKAGE_NAME": memkind_posix_memalign params %p %p %lu %lu\n", kind, memptr, alignment, size);
  } 
#endif

  if (real_memkind_posix_memalign != NULL && canInstrument)
  {
    Backend_Enter_Instrumentation ();
    Probe_memkind_posix_memalign_Entry (get_memkind_partition( kind ), size);
    TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
    res = real_memkind_posix_memalign(kind, memptr, alignment, size);
    if (res == 0)
    {
      Extrae_malloctrace_add (*memptr, size);
    }
    Probe_memkind_posix_memalign_Exit (*memptr);
    Backend_Leave_Instrumentation ();
  }
  else if (real_memkind_posix_memalign != NULL)
  {
    res = real_memkind_posix_memalign(kind, memptr, alignment, size);
  }
  else
  {
    fprintf (stderr, PACKAGE_NAME": memkind_posix_memalign is not hooked! exiting!!\n");
    abort();
  }
  return res;
}

void memkind_free(memkind_t kind, void *ptr)
{
  int canInstrument = EXTRAE_INITIALIZED()                 &&
                      mpitrace_on                          &&
                      Extrae_get_trace_malloc();
  /* Can't be evaluated before because the compiler optimizes the if's clauses, and THREADID calls a null callback if Extrae is not yet initialized */
  if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

  if (real_memkind_free == NULL)
  {
	real_memkind_free = EXTRAE_DL_INIT (__func__);
  }

#if defined(DEBUG)
  if (canInstrument && !__in_free) // fprintf() seems to call free()!
  {
    __in_free = TRUE;
    fprintf (stderr, PACKAGE_NAME": memkind_free is at %p\n", real_memkind_free);
    fprintf (stderr, PACKAGE_NAME": memkind_free params %p %p\n", kind, ptr);
    __in_free = FALSE;
  }
#endif
  
  int present = Extrae_malloctrace_remove (ptr);
  
  if (Extrae_get_trace_malloc_free() && real_memkind_free != NULL && canInstrument && present)
  {
    Backend_Enter_Instrumentation ();
    Probe_memkind_free_Entry (get_memkind_partition( kind ), ptr);
    real_memkind_free (kind, ptr);
    Probe_memkind_free_Exit ();
    Backend_Leave_Instrumentation ();
  }
  else if (real_memkind_free != NULL)
  {
    real_memkind_free (kind, ptr);
  }
  else
  {
    fprintf (stderr, PACKAGE_NAME": memkind_free is not hooked! exiting!!\n");
    abort();
  }
}

#  endif /* HAVE_MEMKIND */

#  if defined(HAVE_OPENMP)
void *
kmpc_malloc( size_t size )
{
	void *res;
	int canInstrument = EXTRAE_INITIALIZED()                 &&
	                    mpitrace_on                          &&
	                    Extrae_get_trace_malloc()            &&
	                    Extrae_get_trace_malloc_allocate()   &&
	                    size >= Extrae_get_trace_malloc_allocate_threshold();
	/* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_kmpc_malloc == NULL)
		real_kmpc_malloc = EXTRAE_DL_INIT (__func__);

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_malloc is at %p\n", real_kmpc_malloc);
		fprintf (stderr, PACKAGE_NAME": kmpc_malloc params %lu\n", size);
	}
#endif

	if (real_kmpc_malloc != NULL && canInstrument)
	{
		/* If we can instrument, simply capture everything we need 
		   and add the pointer to the list of recorded pointers */
		Backend_Enter_Instrumentation ();
		Probe_kmpc_malloc_Entry (size);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_kmpc_malloc (size);
		if (res != NULL)
		{
			Extrae_malloctrace_add (res, size);
		}
		Probe_kmpc_malloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_kmpc_malloc != NULL)
	{
		/* Otherwise, call the original */
		res = real_kmpc_malloc (size);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_malloc is not hooked! exiting!!\n");
		abort();
	}

    return res;

}

void *
kmpc_aligned_malloc( size_t size, size_t alignment )
{
	void *res;
	int canInstrument = EXTRAE_INITIALIZED()                 &&
	                    mpitrace_on                          &&
	                    Extrae_get_trace_malloc()            &&
	                    Extrae_get_trace_malloc_allocate()   &&
	                    size >= Extrae_get_trace_malloc_allocate_threshold();
	/* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_kmpc_aligned_malloc == NULL)
		real_kmpc_aligned_malloc = EXTRAE_DL_INIT (__func__);

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_aligned_malloc is at %p\n", real_kmpc_aligned_malloc);
		fprintf (stderr, PACKAGE_NAME": kmpc_aligned_malloc params %lu\n", size);
	}
#endif

	if (real_kmpc_aligned_malloc != NULL && canInstrument)
	{
		/* If we can instrument, simply capture everything we need 
		   and add the pointer to the list of recorded pointers */
		Backend_Enter_Instrumentation ();
		Probe_kmpc_aligned_malloc_Entry (size, alignment);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_kmpc_aligned_malloc (size, alignment);
		if (res != NULL)
		{
			Extrae_malloctrace_add (res, size);
		}
		Probe_kmpc_aligned_malloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_kmpc_aligned_malloc != NULL)
	{
		/* Otherwise, call the original */
		res = real_kmpc_aligned_malloc (size, alignment);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_malloc is not hooked! exiting!!\n");
		abort();
	}

    return res;

}

void *
kmpc_calloc( size_t nelem, size_t elsize )
{
	void *res;
	int canInstrument = EXTRAE_INITIALIZED()                 &&
	                    mpitrace_on                          &&
	                    Extrae_get_trace_malloc()            &&
	                    Extrae_get_trace_malloc_allocate()   &&
	                    elsize >= Extrae_get_trace_malloc_allocate_threshold();
	/* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_kmpc_calloc == NULL)
		real_kmpc_calloc = EXTRAE_DL_INIT (__func__);

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_calloc is at %p\n", real_kmpc_calloc);
		fprintf (stderr, PACKAGE_NAME": kmpc_calloc params %lu, %lu\n", nelem, elsize);
	}
#endif

	if (real_kmpc_calloc != NULL && canInstrument)
	{
		/* If we can instrument, simply capture everything we need 
		   and add the pointer to the list of recorded pointers */
		Backend_Enter_Instrumentation ();
		Probe_kmpc_calloc_Entry (nelem, elsize);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_kmpc_calloc (nelem, elsize);
		if (res != NULL)
		{
			Extrae_malloctrace_add (res, nelem*elsize);
		}
		Probe_kmpc_calloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_kmpc_calloc != NULL)
	{
		/* Otherwise, call the original */
		res = real_kmpc_calloc (nelem, elsize);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_calloc is not hooked! exiting!!\n");
		abort();
	}

    return res;

}

void *
kmpc_realloc( void *ptr, size_t size )
{
	void *res;
	int canInstrument = EXTRAE_INITIALIZED()                 &&
	                    mpitrace_on                          &&
	                    Extrae_get_trace_malloc()            &&
	                    Extrae_get_trace_malloc_allocate()   &&
	                    size >= Extrae_get_trace_malloc_allocate_threshold();
	/* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_kmpc_realloc == NULL)
		real_kmpc_realloc = EXTRAE_DL_INIT (__func__);

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_realloc is at %p\n", real_kmpc_realloc);
		fprintf (stderr, PACKAGE_NAME": kmpc_realloc params %p, %lu\n", ptr, size);
	}
#endif

	if (real_kmpc_realloc != NULL && canInstrument)
	{
		/* If we can instrument, simply capture everything we need 
		   and add the pointer to the list of recorded pointers */

		int usable_size;


		Backend_Enter_Instrumentation ();
		usable_size = Probe_kmpc_realloc_Entry (ptr, size);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_kmpc_realloc (ptr, size);
		if (res != NULL)
		{
			Extrae_malloctrace_replace (ptr, res, size);
		}
		Probe_kmpc_realloc_Exit (res, usable_size);
		Backend_Leave_Instrumentation ();
	}
	else if (real_kmpc_realloc != NULL)
	{
		/* Otherwise, call the original */
		res = real_kmpc_realloc (ptr, size);
		// We may need to remove the previous pointer
		Extrae_malloctrace_remove (ptr);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_realloc is not hooked! exiting!!\n");
		abort();
	}

    return res;

}

void
kmpc_free ( void *ptr )
{
	int canInstrument = EXTRAE_INITIALIZED()                 &&
	                    mpitrace_on                          &&
	                    Extrae_get_trace_malloc();
	/* Can't be evaluated before because the compiler optimizes the if's clauses,
	 * and THREADID calls a null callback if Extrae is not yet initialized */
	if (canInstrument) canInstrument = !Backend_inInstrumentation(THREADID);

	if (real_kmpc_free == NULL)
		real_kmpc_free = EXTRAE_DL_INIT (__func__);

#if defined(DEBUG)
	if (canInstrument && !__in_free) // fprintf() seems to call free()!
	{
		__in_free = TRUE;
		fprintf (stderr, PACKAGE_NAME": kmpc_free is at %p\n", real_kmpc_free);
		fprintf (stderr, PACKAGE_NAME": kmpc_free params %p\n", ptr);
		__in_free = FALSE;
	}
#endif

	int present = Extrae_malloctrace_remove (ptr);

	if (Extrae_get_trace_malloc_free() &&
	    real_kmpc_free != NULL         &&
			canInstrument                  &&
			present)
	{
		/* If we can instrument, simply capture everything we need and
		   remove the pointer from the list */
		Backend_Enter_Instrumentation ();
		Probe_kmpc_free_Entry (ptr);
		real_kmpc_free (ptr);
		Probe_kmpc_free_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_kmpc_free != NULL)
	{
		/* Otherwise, call the original */
		real_kmpc_free (ptr);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": kmpc_free is not hooked! exiting!!\n");
		abort();
	}
}
#  endif /* HAVE_OPENMP */

# endif /* -DPIC */

#endif /* INSTRUMENT_DYNAMIC_MEMORY */
