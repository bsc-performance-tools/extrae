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

#include "omp-common_f.h"
#include "omp-probe.h"

// #define DEBUG

/*
 * In case the constructor initialization didn't trigger
 * or the symbols couldn't be found, retry hooking.
 */
#define RECHECK_INIT_F(real_fn_ptr)                                            \
{                                                                              \
	if (real_fn_ptr == NULL)                                               \
	{                                                                      \
		fprintf (stderr, PACKAGE_NAME                                  \
		    ": WARNING! %s is a NULL pointer. "                        \
		    "Did the initialization of this module trigger? "          \
		    "Retrying initialization...\n", #real_fn_ptr);             \
		omp_common_get_hook_points_f(TASKID);                          \
	}                                                                      \
}

#if defined(PIC)
void (*omp_set_num_threads__real)(int*) = NULL;
void (*omp_set_num_threads_8__real)(int*) = NULL;
void (*omp_set_lock__real)(omp_lock_t *) = NULL;
void (*omp_unset_lock__real)(omp_lock_t *) = NULL;
#endif /* PIC */

void omp_common_get_hook_points_f (int rank)
{
	UNREFERENCED_PARAMETER(rank);

#if defined(PIC)

	/* Obtain @ for omp_set_num_threads_ */
	omp_set_num_threads__real =
	    (void(*)(int*)) dlsym (RTLD_NEXT, "omp_set_num_threads_");

	/* Obtain @ for omp_set_num_threads_8_ */
	omp_set_num_threads_8__real =
	    (void(*)(int*)) dlsym (RTLD_NEXT, "omp_set_num_threads_8_");

	/* Obtain @ for omp_set_lock_ */
	omp_set_lock__real =
	    (void(*)(omp_lock_t*)) dlsym (RTLD_NEXT, "omp_set_lock_");

	/* Obtain @ for omp_unset_lock_ */
	omp_unset_lock__real =
	    (void(*)(omp_lock_t*)) dlsym (RTLD_NEXT, "omp_unset_lock_");

#endif /* PIC */
}

/******************************************************************************\
 *                                                                            * 
 *                        FORTRAN WRAPPERS                                    *
 * 																			  *
 * IMPORTANT: Download GCC and review in libgomp/fortran.c the different      *
 * symbols to implement the wrappers correctly.                        		  *
 *                                                                            * 
\******************************************************************************/

#if defined(PIC)

void
omp_set_num_threads_(int *num_threads)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME
	    ": THREAD %d: omp_set_num_threads_ starts (real=%p) params=(%d)\n",
	    THREADID, omp_set_num_threads__real, *num_threads);
#endif

	RECHECK_INIT_F(omp_set_num_threads__real);

	int canInstrument = EXTRAE_INITIALIZED() &&
						omp_set_num_threads__real != NULL;

	if (canInstrument && !Backend_inInstrumentation(THREADID))
	{
		/*
		 * Change number of threads only if in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(*num_threads);

		Backend_Enter_Instrumentation();
		Probe_OpenMP_SetNumThreads_Entry(*num_threads);
		omp_set_num_threads__real(num_threads);
		Probe_OpenMP_SetNumThreads_Exit();
		Backend_Leave_Instrumentation();
	}
	else if (omp_set_num_threads__real != NULL)
	{
		omp_set_num_threads__real(num_threads);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME
		    ": ERROR! omp_set_num_threads_ is not hooked! Exiting!!\n");
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME
	    ": THREAD %d: omp_set_num_threads_ ends\n", THREADID);
#endif
}

void
omp_set_num_threads_8_(int *num_threads)
{
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME
	    ": THREAD %d: omp_set_num_threads_8_ starts (real=%p) params=(%d)\n",
	    THREADID, omp_set_num_threads_8__real, *num_threads);
#endif

	RECHECK_INIT_F(omp_set_num_threads_8__real);

	int canInstrument = EXTRAE_INITIALIZED() &&
						omp_set_num_threads_8__real != NULL;

	if (canInstrument && !Backend_inInstrumentation(THREADID))
	{
		/*
		 * Change number of threads only if in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(*num_threads);

		Backend_Enter_Instrumentation();
		Probe_OpenMP_SetNumThreads_Entry(*num_threads);
		omp_set_num_threads_8__real(num_threads);
		Probe_OpenMP_SetNumThreads_Exit();
		Backend_Leave_Instrumentation();
	}
	else if (omp_set_num_threads_8__real != NULL)
	{
		omp_set_num_threads_8__real(num_threads);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME
		    ": ERROR! omp_set_num_threads_8_ is not hooked! Exiting!!\n");
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME
	    ": THREAD %d: omp_set_num_threads_8_ ends\n", THREADID);
#endif
}

void omp_set_lock_ (omp_lock_t *lock)
{
	void *lock_ptr = (void *)lock;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_lock_ starts (real=%p) params=(%p)\n", THREADID, omp_set_lock__real, lock_ptr);
#endif

	RECHECK_INIT_F(omp_set_lock__real);
	
	int canInstrument = EXTRAE_INITIALIZED()	&&
						omp_set_lock__real != NULL;

	if (canInstrument && !Backend_inInstrumentation(THREADID))
	{
		Backend_Enter_Instrumentation ();
		Probe_OpenMP_Named_Lock_Entry();
		omp_set_lock__real (lock_ptr);
		Probe_OpenMP_Named_Lock_Exit(lock_ptr);
		Backend_Leave_Instrumentation ();
	}
	else if (omp_set_lock__real != NULL)
	{
		omp_set_lock__real (lock);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! omp_set_lock_ is not hooked! Exiting!!\n");
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_set_lock_ ends\n", THREADID);
#endif
}

void omp_unset_lock_ (omp_lock_t *lock)
{
	void *lock_ptr = (void *)lock;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_unset_lock_ starts (real=%p) params=(%p)\n", THREADID, omp_unset_lock__real, lock_ptr);
#endif

	RECHECK_INIT_F(omp_unset_lock__real);

	int canInstrument = EXTRAE_INITIALIZED()	&&
						omp_unset_lock__real != NULL;

	if (canInstrument && !Backend_inInstrumentation(THREADID))
	{
		Backend_Enter_Instrumentation ();
		Probe_OpenMP_Named_Unlock_Entry(lock_ptr);
		omp_unset_lock__real (lock_ptr);
		Probe_OpenMP_Named_Unlock_Exit();
		Backend_Leave_Instrumentation ();
	}
	else if (omp_unset_lock__real != NULL)
	{
		omp_unset_lock__real (lock);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! omp_unset_lock_ is not hooked! Exiting!!\n");
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: omp_unset_lock_ ends\n", THREADID);
#endif
}

#endif /* PIC */
