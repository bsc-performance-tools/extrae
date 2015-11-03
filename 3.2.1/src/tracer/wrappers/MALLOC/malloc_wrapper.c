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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.4/src/tracer/wrappers/OMP/omp_wrapper.c $
 | @last_commit: $Date: 2013-09-06 14:39:32 +0200 (Fri, 06 Sep 2013) $
 | @version:     $Revision: 2098 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: omp_wrapper.c 2098 2013-09-06 12:39:32Z harald $";

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

#include "wrapper.h"
#include "trace_macros.h"
#include "malloc_probe.h"

#if defined(INSTRUMENT_DYNAMIC_MEMORY)

/*
	This wrappers can only be compiled if the file is being compiled to
	generate a shared library (-DPIC)
*/

static void* (*real_malloc)(size_t) = NULL;
static void (*real_free)(void *) = NULL;
/* static void* (*real_calloc)(size_t, size_t) = NULL; */
static void* (*real_realloc)(void*, size_t) = NULL;

void Extrae_malloctrace_init (void)
{
# if defined(PIC) /* This is only available for .so libraries */
	real_free = (void(*)(void*)) dlsym (RTLD_NEXT, "free");
	real_malloc = (void*(*)(size_t)) dlsym (RTLD_NEXT, "malloc");
	/* real_calloc = (void*(*)(size_t, size_t)) dlsym (RTLD_NEXT, "calloc"); */
	real_realloc = (void*(*)(void*, size_t)) dlsym (RTLD_NEXT, "realloc");
# else
	fprintf (stderr, PACKAGE_NAME": Warning! dynamic memory instrumentation requires linking with shared library!\n");
# endif
}

//#define DEBUG

/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/

#define TRACE_DYNAMIC_MEMORY_CALLER_IS_ENABLED \
 (Trace_Caller_Enabled[CALLER_DYNAMIC_MEMORY])

#define TRACE_DYNAMIC_MEMORY_CALLER(evttime,offset) \
{ \
	if (TRACE_DYNAMIC_MEMORY_CALLER_IS_ENABLED) \
		trace_callers (evttime, offset, CALLER_DYNAMIC_MEMORY); \
}

# if defined(PIC) /* This is only available for .so libraries */
void *malloc (size_t s)
{
	void *res;
	int canInstrument = !Backend_inInstrumentation(THREADID) && 
	  mpitrace_on &&
	  Extrae_get_trace_malloc() &&
	  Extrae_get_trace_malloc_allocate() &&
	  s >= Extrae_get_trace_malloc_allocate_threshold();

	if (real_malloc == NULL)
		Extrae_malloctrace_init ();

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": malloc is at %p\n", real_malloc);
		fprintf (stderr, PACKAGE_NAME": malloc params %lu\n", s);
	}
#endif

	if (real_malloc != NULL && canInstrument)
	{
		Backend_Enter_Instrumentation (2);
		Probe_Malloc_Entry (s);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_malloc (s);
		Probe_Malloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_malloc != NULL)
	{
		res = real_malloc (s);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": malloc is not hooked! exiting!!\n");
		abort();
	}

    return res;
}

#if defined(DEBUG)
static int __in_free = FALSE;
#endif
void free (void *p)
{
	int canInstrument = !Backend_inInstrumentation(THREADID) && 
	  mpitrace_on &&
	  Extrae_get_trace_malloc();

	if (real_free == NULL)
		Extrae_malloctrace_init ();

#if defined(DEBUG)
	if (canInstrument && !__in_free) // fprintf() seems to call free()!
	{
		__in_free = TRUE;
		fprintf (stderr, PACKAGE_NAME": free is at %p\n", real_free);
		fprintf (stderr, PACKAGE_NAME": free params %p\n", p);
		__in_free = FALSE;
	}
#endif

	if (Extrae_get_trace_malloc_free() && real_free != NULL && canInstrument)
	{
		Backend_Enter_Instrumentation (2);
		Probe_Free_Entry (p);
		real_free (p);
		Probe_Free_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_free != NULL)
	{
		real_free (p);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": free is not hooked! exiting!!\n");
		abort();
	}
}

#if 0
/* Unfortunately, calloc seems to be invoked if dlsym fails and generates an
an infinite loop of recursive calls to calloc */
void *calloc (size_t s1, size_t s2)
{
	void *res;
	int canInstrument = !Backend_inInstrumentation(THREADID) && 
	  mpitrace_on &&
	  Extrae_get_trace_malloc();

	if (real_calloc == NULL)
		Extrae_malloctrace_init ();

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": calloc is at %p\n", real_calloc);
	fprintf (stderr, PACKAGE_NAME": calloc params %u %u\n", s1, s2);
#endif

	if (real_calloc != NULL && canInstrument)
	{
		Backend_Enter_Instrumentation (2);
		Probe_Calloc_Entry (s1, s2);
		res = real_calloc (s1, s2);
		Probe_Calloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_calloc != NULL && !canInstrument)
	{
		res = real_calloc (s1, s2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": calloc is not hooked! exiting!!\n");
		abort();
	}

    return res;
}
#endif

void *realloc (void *p, size_t s)
{
	void *res;
	int canInstrument = !Backend_inInstrumentation(THREADID) && 
	  mpitrace_on &&
	  Extrae_get_trace_malloc() &&
	  Extrae_get_trace_malloc_allocate() &&
	  s >= Extrae_get_trace_malloc_allocate_threshold();

	if (real_realloc == NULL)
		Extrae_malloctrace_init ();

#if defined(DEBUG)
	if (canInstrument)
	{
		fprintf (stderr, PACKAGE_NAME": realloc is at %p\n", real_realloc);
		fprintf (stderr, PACKAGE_NAME": realloc params %p %lu\n", p, s);
	}
#endif

	if (real_realloc != NULL && canInstrument)
	{
		Backend_Enter_Instrumentation (2);
		Probe_Realloc_Entry (p, s);
		TRACE_DYNAMIC_MEMORY_CALLER(LAST_READ_TIME, 3);
		res = real_realloc (p, s);
		Probe_Realloc_Exit (res);
		Backend_Leave_Instrumentation ();
	}
	else if (real_realloc != NULL)
	{
		res = real_realloc (p, s);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": realloc is not hooked! exiting!!\n");
		abort();
	}

    return res;
}

# endif /* -DPIC */

#endif /* INSTRUMENT_DYNAMIC_MEMORY */
