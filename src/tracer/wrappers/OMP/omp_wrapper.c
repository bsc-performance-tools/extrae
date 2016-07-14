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
#include "omp_probe.h"
#include "omp_wrapper.h"

#include "ibm-xlsmp-1.6.h"
#if defined(GNU_OPENMP_4_2)
# include "gnu-libgomp-4.2.h"
#elif defined(GNU_OPENMP_4_9)
# include "gnu-libgomp-4.9.h"
#endif
#include "intel-kmpc-11.h"

//#define DEBUG

#if defined(STANDALONE)
void __attribute__ ((constructor)) extrae_openmp_setup(void)
{
  fprintf(stderr, "[DEBUG-OPENMP] Registering module init=%p\n", Extrae_OpenMP_init);
  Extrae_RegisterModule(OPENMP_MODULE, Extrae_OpenMP_init, NULL);
}
#endif

#if defined(PIC)
static int (*omp_get_thread_num_real)(void) = NULL;
static void (*omp_set_lock_real)(void *) = NULL;
static void (*omp_unset_lock_real)(void *) = NULL;
static void (*omp_set_num_threads_real)(int) = NULL;
#endif /* PIC */

static void common_GetOpenMPHookPoints (int rank)
{
	UNREFERENCED_PARAMETER(rank);

#if defined(PIC)
	/* Obtain @ for omp_set_lock */
	omp_get_thread_num_real =
		(int(*)(void)) dlsym (RTLD_NEXT, "omp_get_thread_num");

	/* Obtain @ for omp_set_lock */
	omp_set_lock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "omp_set_lock");

	/* Obtain @ for omp_unset_lock */
	omp_unset_lock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "omp_unset_lock");

	/* Obtain @ for omp_set_num_threads */
	omp_set_num_threads_real =
		(void(*)(int)) dlsym (RTLD_NEXT, "omp_set_num_threads");
#endif /* PIC */
}

/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/

#if defined(PIC)
int omp_get_thread_num (void)
{
	int res;
	static int shown = FALSE;

        if (omp_get_thread_num_real == NULL)
	{
		common_GetOpenMPHookPoints(0);
	}

	if (omp_get_thread_num_real != NULL)
	{
		res = omp_get_thread_num_real();
	}
	else
	{
		if (!shown)
		{
			fprintf (stderr,
			  PACKAGE_NAME": Caution! Caution! Caution! Caution! -------------------- \n"
			  PACKAGE_NAME":\n"
			  PACKAGE_NAME": You have ended executing Extrae's omp_get_thread_num weak symbol!\n"
			  PACKAGE_NAME": That's likely to happen when you instrument your application using OpenMP\n"
			  PACKAGE_NAME": instrumentation, but your application is not compiled/linked against OpenMP\n"
			  PACKAGE_NAME":\n"
			  PACKAGE_NAME": Caution! Caution! Caution! Caution! -------------------- \n");
			shown = TRUE;
		}
		res = 0;
	}
	return res;
}
#endif

#if defined(PIC)
void omp_set_lock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": omp_set_lock is at %p\n", omp_set_lock_real);
	fprintf (stderr, PACKAGE_NAME": omp_set_lock params %p\n", p1);
#endif

	if (omp_set_lock_real != NULL && EXTRAE_INITIALIZED())
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Lock_Entry();
		omp_set_lock_real (p1);
		Probe_OpenMP_Named_Lock_Exit(p1);
		Backend_Leave_Instrumentation ();
	}
	else if (omp_set_lock_real != NULL)
	{
		omp_set_lock_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": omp_set_lock is not hooked! exiting!!\n");
		exit (0);
	}
}

void omp_unset_lock (int *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": omp_unset_lock is at %p\n", omp_unset_lock_real);
	fprintf (stderr, PACKAGE_NAME": omp_unset_lock params %p\n", p1);
#endif

	if (omp_unset_lock_real != NULL && EXTRAE_INITIALIZED())
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Unlock_Entry(p1);
		omp_unset_lock_real (p1);
		Probe_OpenMP_Named_Unlock_Exit();
		Backend_Leave_Instrumentation ();
	}
	else if (omp_unset_lock_real != NULL)
	{
		omp_unset_lock_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": omp_unset_lock is not hooked! exiting!!\n");
		exit (0);
	}
}

void omp_set_num_threads (int p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": omp_set_num_threads is at %p\n", omp_set_num_threads_real);
	fprintf (stderr, PACKAGE_NAME": omp_set_num_threads params %d\n", p1);
#endif

	if (omp_set_num_threads_real != NULL && EXTRAE_INITIALIZED())
	{
		Backend_ChangeNumberOfThreads (p1);

		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_SetNumThreads_Entry (p1);
		omp_set_num_threads_real (p1);
		Probe_OpenMP_SetNumThreads_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (omp_set_num_threads_real != NULL)
	{
		omp_set_num_threads_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": omp_set_num_threads is not hooked! exiting!!\n");
		exit (0);
	}
}
#endif /* PIC */

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

void Extrae_OpenMP_init(int me)
{
	UNREFERENCED_PARAMETER(me);

#if defined(PIC)
	int hooked = FALSE;

# if defined(OS_LINUX) && defined(ARCH_PPC) && defined(IBM_OPENMP)
	/*
	 * On PPC systems, check first for IBM XL runtime, if we don't find any
	 * symbol, check for GNU then 
	 */
	hooked = ibm_xlsmp_1_6_hook_points(0);
	if (!hooked)
	{
		fprintf (stdout, PACKAGE_NAME": ATTENTION! Application seems not to be linked with IBM XL OpenMP runtime!\n");
	}
# endif /* OS_LINUX && ARCH_PPC && IBM_OPENMP */

# if defined(INTEL_OPENMP)
	if (!hooked)
	{
		hooked = intel_kmpc_11_hook_points(0);
		if (!hooked)
		{
			fprintf (stdout, PACKAGE_NAME": ATTENTION! Application seems not to be linked with Intel KAP OpenMP runtime!\n");
		}
	}
# endif /* INTEL_OPENMP */

# if defined(GNU_OPENMP)
	if (!hooked)
	{
# if defined(GNU_OPENMP_4_2)
		hooked = gnu_libgomp_4_2_hook_points(0);
# elif defined(GNU_OPENMP_4_9)
		hooked = gnu_libgomp_4_9_hook_points(0);
# else
#  error "Unsupported version of libgomp!"
# endif 
		if (!hooked)
		{
			fprintf (stdout, PACKAGE_NAME": ATTENTION! Application seems not to be linked with GNU OpenMP runtime!\n");
		}
	}
# endif /* GNU_OPENMP */

	/* 
	 * If we hooked any compiler-specific routines, just hook for the 
	 * common OpenMP routines 
	 */

	if (hooked)
		common_GetOpenMPHookPoints(0);

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

