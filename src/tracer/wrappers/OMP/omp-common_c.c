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

#include "omp-common_c.h"
#include "omp-probe.h"

// #define DEBUG

/*
 * In case the constructor initialization didn't trigger
 * or the symbols couldn't be found, retry hooking.
 */
#define RECHECK_INIT_C(real_fn_ptr)                                            \
{                                                                              \
	if (real_fn_ptr == NULL)                                               \
	{                                                                      \
		fprintf (stderr, PACKAGE_NAME                                  \
		    ": WARNING! %s is a NULL pointer. "                        \
		    "Did the initialization of this module trigger? "          \
		    "Retrying initialization...\n", #real_fn_ptr);             \
		omp_common_get_hook_points_c(TASKID);                          \
	}                                                                      \
}

#if defined(PIC)
int (*omp_get_thread_num_real)(void) = NULL;
void (*omp_set_num_threads_real)(int) = NULL;
void (*omp_set_lock_real)(omp_lock_t *) = NULL;
void (*omp_unset_lock_real)(omp_lock_t *) = NULL;
#endif /* PIC */

void omp_common_get_hook_points_c (int rank)
{
	UNREFERENCED_PARAMETER(rank);

#if defined(PIC)
	/* Obtain @ for omp_get_thread_num_real */
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
 *                              C WRAPPERS                                    * 
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

  RECHECK_INIT_C(omp_get_thread_num_real);

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

void
omp_set_num_threads(int num_threads)
{
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME
	    ": THREAD %d: omp_set_num_threads starts (real=%p) params=(%d)\n",
	    THREADID, omp_set_num_threads_real, num_threads);
#endif

	RECHECK_INIT_C(omp_set_num_threads_real);

	int canInstrument = EXTRAE_INITIALIZED() &&
						omp_set_num_threads_real != NULL;

	if (canInstrument && !Backend_inInstrumentation(THREADID))
	{
		/*
		 * Change number of threads only if in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Backend_Enter_Instrumentation();
		Probe_OpenMP_SetNumThreads_Entry(num_threads);
		omp_set_num_threads_real(num_threads);
		Probe_OpenMP_SetNumThreads_Exit();
		Backend_Leave_Instrumentation();
	}
	else if (omp_set_num_threads_real != NULL)
	{
		omp_set_num_threads_real(num_threads);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME
		    ": ERROR! omp_set_num_threads is not hooked! Exiting!!\n");
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d: omp_set_num_threads ends\n",
	    THREADID);
#endif
}

void omp_set_lock (omp_lock_t *lock)
{
	void *lock_ptr = (void *)lock;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_lock starts (real=%p) params=(%p)\n", THREADID, omp_set_lock_real, lock_ptr);
#endif

	RECHECK_INIT_C(omp_set_lock_real);
	
	int canInstrument = EXTRAE_INITIALIZED()	&&
						omp_set_lock_real != NULL;

	if (canInstrument && !Backend_inInstrumentation(THREADID))
	{
		Backend_Enter_Instrumentation ();
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

	RECHECK_INIT_C(omp_unset_lock_real);

	int canInstrument = EXTRAE_INITIALIZED()	&&
						omp_unset_lock_real != NULL;

	if (canInstrument && !Backend_inInstrumentation(THREADID))
	{
		Backend_Enter_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_unset_lock ends\n", THREADID);
#endif
}

#endif /* PIC */
