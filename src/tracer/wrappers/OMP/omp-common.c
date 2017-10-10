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

#ifdef HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef  __USE_GNU
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "wrapper.h"
#include "trace_macros.h"
#include "omp-probe.h"
#include "omp-common.h"
#include "omp-events.h"

#include "ibm-xlsmp-1.6.h"
#include "gnu-libgomp.h"
#include "intel-kmpc-11.h"

//#define DEBUG

/*
 * In case the constructor initialization didn't trigger
 * or the symbols couldn't be found, retry hooking.
 */
#define RECHECK_INIT(real_fn_ptr)                                      \
{                                                                      \
  if (real_fn_ptr == NULL)                                             \
  {                                                                    \
    fprintf (stderr, PACKAGE_NAME": WARNING! %s is a NULL pointer. "   \
                     "Did the initialization of this module trigger? " \
                     "Retrying initialization...\n", #real_fn_ptr);    \
    omp_common_get_hook_points(TASKID);                                \
  }                                                                    \
}                                                                                

#if defined(PIC)
static int (*omp_get_thread_num_real)(void) = NULL;
static void (*omp_set_num_threads_real)(int) = NULL;
static void (*omp_set_lock_real)(omp_lock_t *) = NULL;
static void (*omp_unset_lock_real)(omp_lock_t *) = NULL;
#endif /* PIC */

/******************************************************************************\
 *                                                                            * 
 *                                  HELPERS                                   * 
 *                                                                            * 
 ****************************************************************************** 
 * The following helper structures are used to pass information from the      *
 * master thread that opens a parallel, to the worker threads that run the    *
 * parallel constructs (for, do...). These is a generic structure that holds  *
 * data for the different runtimes, each runtime fills their respective       *
 * fields.                                                                    *
\******************************************************************************/

/*
 * A matrix indexed by thread id and nesting level to store helper data 
 */
struct thread_helper_t **__omp_nested_storage = NULL;

/**
 * allocate_nested_helpers
 *
 * Allocates a matrix indexed by thread id and nesting level so that master 
 * threads can store helper data that needs to be later retrieved by worker 
 * threads in a deeper nesting level.
 */
void allocate_nested_helpers()
{
	int i = 0, j = 0;

	if (__omp_nested_storage == NULL)
	{
		__omp_nested_storage = (struct thread_helper_t **)malloc(omp_get_max_threads() * sizeof(struct thread_helper_t *));
		for (i=0; i<omp_get_max_threads(); i++)
		{
			__omp_nested_storage[i] = (struct thread_helper_t *)malloc(MAX_NESTING_LEVEL * sizeof(struct thread_helper_t));
			for (j=0; j<MAX_NESTING_LEVEL; j++)
			{
				__omp_nested_storage[i][j].par_uf = NULL;
			}
		}
	}
}

/**
 * get_thread_helper
 *
 * @return the adress of the helper to store data for the current thread 
 * in the current nesting level.
 */
struct thread_helper_t * get_thread_helper()
{
	int thread_id = THREADID;
	int nesting_level = omp_get_level();

	return &(__omp_nested_storage[thread_id][nesting_level]);
}

/**
 * get_parent_thread_helper
 *
 * @return the address of the helper that stores data for the current's thread
 * parent in the previous nesting level.
 */
struct thread_helper_t * get_parent_thread_helper()
{
	int nesting_level = omp_get_level();
	int parent_level = nesting_level - 1;
	int parent_id = omp_get_ancestor_thread_num(parent_level);

	return &(__omp_nested_storage[parent_id][parent_level]);
}


static void omp_common_get_hook_points (int rank)
{
	UNREFERENCED_PARAMETER(rank);

#if defined(PIC)
	/* Obtain @ for omp_set_lock */
	omp_get_thread_num_real =
		(int(*)(void)) dlsym (RTLD_NEXT, "omp_get_thread_num");

	/* Obtain @ for omp_set_num_threads */
	omp_set_num_threads_real =
		(void(*)(int)) dlsym (RTLD_NEXT, "omp_set_num_threads");

	/* Obtain @ for omp_set_lock */
	omp_set_lock_real =
		(void(*)(omp_lock_t*)) dlsym (RTLD_NEXT, "omp_set_lock");

	/* Obtain @ for omp_unset_lock */
	omp_unset_lock_real =
		(void(*)(omp_lock_t*)) dlsym (RTLD_NEXT, "omp_unset_lock");

#endif /* PIC */
}

/******************************************************************************\
 *                                                                            * 
 *                                WRAPPERS                                    * 
 *                                                                            * 
\******************************************************************************/

#if defined(PIC)

int omp_get_thread_num (void)
{
	static int shown = FALSE;
	int res = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": omp_get_thread_num starts (real=%p) params=(void)\n", omp_get_thread_num_real);
#endif

  RECHECK_INIT(omp_get_thread_num_real);

	if (omp_get_thread_num_real != NULL)
	{
		res = omp_get_thread_num_real();
	}
	else
	{
		res = 0;

		if (!shown)
		{
			fprintf (stderr,
			  PACKAGE_NAME": WARNING! You have ended executing Extrae's omp_get_thread_num weak symbol! "
			  "That's likely to happen when you load the tracing library for OpenMP, "
			  "but your application is not compiled/linked against OpenMP.\n" );
			shown = TRUE;
		}
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_get_thread_num ends\n", res);
#endif

	return res;
}

void omp_set_num_threads (int num_threads)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_num_threads starts (real=%p) params=(%d)\n", THREADID, omp_set_num_threads_real, num_threads);
#endif

	RECHECK_INIT(omp_set_num_threads_real);

	if (omp_set_num_threads_real != NULL && EXTRAE_INITIALIZED())
	{
		Backend_ChangeNumberOfThreads (num_threads);

		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_SetNumThreads_Entry (num_threads);
		omp_set_num_threads_real (num_threads);
		Probe_OpenMP_SetNumThreads_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (omp_set_num_threads_real != NULL)
	{
		omp_set_num_threads_real (num_threads);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! omp_set_num_threads is not hooked! Exiting!!\n");
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_num_threads ends\n", THREADID);
#endif
}

void omp_set_lock (omp_lock_t *lock)
{
	void *lock_ptr = (void *)lock;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_lock starts (real=%p) params=(%p)\n", THREADID, omp_set_lock_real, lock_ptr);
#endif

	RECHECK_INIT(omp_set_lock_real);

	if (omp_set_lock_real != NULL && EXTRAE_INITIALIZED())
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Lock_Entry();
		omp_set_lock_real (lock_ptr);
		Probe_OpenMP_Named_Lock_Exit(lock_ptr);
		Backend_Leave_Instrumentation ();
	}
	else if (omp_set_lock_real != NULL)
	{
		omp_set_lock_real (lock);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! omp_set_lock is not hooked! Exiting!!\n");
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_lock ends\n", THREADID);
#endif
}

void omp_unset_lock (omp_lock_t *lock)
{
	void *lock_ptr = (void *)lock;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_unset_lock starts (real=%p) params=(%p)\n", THREADID, omp_unset_lock_real, lock_ptr);
#endif

	RECHECK_INIT(omp_unset_lock_real);

	if (omp_unset_lock_real != NULL && EXTRAE_INITIALIZED())
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Unlock_Entry(lock_ptr);
		omp_unset_lock_real (lock_ptr);
		Probe_OpenMP_Named_Unlock_Exit();
		Backend_Leave_Instrumentation ();
	}
	else if (omp_unset_lock_real != NULL)
	{
		omp_unset_lock_real (lock);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! omp_unset_lock is not hooked! Exiting!!\n");
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_unlock ends\n", THREADID);
#endif
}

#endif /* PIC */

/******************************************************************************\
 *                                                                            * 
 *                             INITIALIZATIONS                                * 
 *                                                                            * 
\******************************************************************************/

#if defined(STANDALONE)
void __attribute__ ((constructor)) extrae_openmp_setup(void)
{
  fprintf(stderr, "[DEBUG-OPENMP] Registering module init=%p\n", Extrae_OpenMP_init);
  Extrae_RegisterModule(OPENMP_MODULE, Extrae_OpenMP_init, NULL);
}
#endif

#if defined(STANDALONE)
static int getnumProcessors (void)
{
	int numProcessors = 0;

#if HAVE_SYSCONF
	numProcessors = (int) sysconf (_SC_NPROCESSORS_CONF);
	if (-1 == numProcessors)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot determine number of configured processors using sysconf\n");
		exit (-1);
	}
#else
# error "Cannot determine number of processors"
#endif

	return numProcessors;
}
#endif

/**
 * Extrae_OpenMP_init
 * 
 * Main initialization for the OpenMP instrumentation module. 
 * Detects the runtime that is present (IBM, Intel, GNU...) and loads
 * specific hooks for the present runtime.
 * Also loads some common hooks for basic OpenMP routines available in 
 * all runtimes.
 */
void Extrae_OpenMP_init(int me)
{
	UNREFERENCED_PARAMETER(me);

#if defined(PIC)
	int ibm_hooked = FALSE;
	int intel_hooked = FALSE;
	int gnu_hooked = FALSE;
	int hooked = 0;

	allocate_nested_helpers();

# if defined(OS_LINUX) && defined(ARCH_PPC) && defined(IBM_OPENMP)
	/*
	 * On PPC systems, check first for IBM XL runtime, if we don't find any
	 * symbol, check for GNU then 
	 */
	ibm_hooked = _extrae_ibm_xlsmp_init(0);
# endif /* OS_LINUX && ARCH_PPC && IBM_OPENMP */

# if defined(INTEL_OPENMP)
	intel_hooked = _extrae_intel_kmpc_init(0);
# endif /* INTEL_OPENMP */

# if defined(GNU_OPENMP)
	gnu_hooked = _extrae_gnu_libgomp_init(0);
# endif /* GNU_OPENMP */

	hooked = ibm_hooked + intel_hooked + gnu_hooked;

	if (hooked > 0) {
		fprintf (stdout, PACKAGE_NAME": Detected and hooked OpenMP runtime:%s%s%s\n",
		         ibm_hooked?" [IBM XLSMP]":"",
		         intel_hooked?" [Intel KMPC]":"",
		         gnu_hooked?" [GNU GOMP]":"");

		/* 
		* If we hooked any compiler-specific routines, just hook for the 
		* common OpenMP routines 
		*/

		omp_common_get_hook_points(0);
	} else {
		fprintf (stdout, PACKAGE_NAME": Warning! You have loaded an OpenMP tracing library but the application seems not to be linked with an OpenMP runtime. Did you compile with the proper flags? (-fopenmp, -openmp, ...).\n");
	}

#else  /* PIC */

	fprintf (stderr, PACKAGE_NAME": Warning! OpenMP instrumentation requires linking with shared library!\n");

#endif /* PIC */

#if defined(STANDALONE)
	int numProcessors = 0;
	char *new_num_omp_threads_clause = NULL;
	char *omp_value = NULL;
	
	/* 
	 * Obtain the number of runnable threads in this execution.
	 * Just check for OMP_NUM_THREADS env var (if this compilation
	 * allows instrumenting OpenMP 
	 */
	numProcessors = getnumProcessors();
	
	new_num_omp_threads_clause = (char*) malloc ((strlen("OMP_NUM_THREADS=xxxx")+1)*sizeof(char));
	if (NULL == new_num_omp_threads_clause)
	{
		fprintf (stderr, PACKAGE_NAME": Unable to allocate memory for tentative OMP_NUM_THREADS\n");
		exit (-1);
	}
	if (numProcessors >= 10000) /* xxxx in new_omp_threads_clause -> max 9999 */
	{
		fprintf (stderr, PACKAGE_NAME": Insufficient memory allocated for tentative OMP_NUM_THREADS\n");
		exit (-1);
	}
	
	sprintf (new_num_omp_threads_clause, "OMP_NUM_THREADS=%d\n", numProcessors);
	omp_value = getenv ("OMP_NUM_THREADS");
	if (omp_value)
	{
		int num_of_threads = atoi (omp_value);
		if (num_of_threads != 0)
		{
			Extrae_core_set_maximum_threads( num_of_threads );
			Extrae_core_set_current_threads( num_of_threads );
			if (me == 0)
			{
				fprintf (stdout, PACKAGE_NAME": OMP_NUM_THREADS set to %d\n", num_of_threads);
			}
		}
		else
		{
			if (me == 0)
			{
				fprintf (stderr,
				PACKAGE_NAME": OMP_NUM_THREADS is mandatory for this tracing library!\n"\
				PACKAGE_NAME": Setting OMP_NUM_THREADS to %d\n", numProcessors);
			}
			putenv (new_num_omp_threads_clause);
			Extrae_core_set_maximum_threads( numProcessors );
			Extrae_core_set_current_threads( numProcessors );
		}
	}
	else
	{
		if (me == 0)
		{
			fprintf (stderr,
			PACKAGE_NAME": OMP_NUM_THREADS is mandatory for this tracing library!\n"\
			PACKAGE_NAME": Setting OMP_NUM_THREADS to %d\n", numProcessors);
		}
		putenv (new_num_omp_threads_clause);
		Extrae_core_set_maximum_threads( numProcessors );
		Extrae_core_set_current_threads( numProcessors );
	}
#endif /* STANDALONE */
}
