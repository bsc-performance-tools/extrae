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
#include "omp-common_c.h"
#include "omp-common_f.h"
#include "omp-events.h"

#include "ibm-xlsmp-1.6.h"
#include "gnu-libgomp.h"
#include "intel-kmpc-11.h"

// #define DEBUG


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
#if defined(PIC)
	int ibm_hooked = FALSE;
	int intel_hooked = FALSE;
	int gnu_hooked = FALSE;
	int hooked = 0;

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
		if (me == 0)
		{
			fprintf (stdout, PACKAGE_NAME": Detected and hooked OpenMP runtime:%s%s%s\n",
			  ibm_hooked?" [IBM XLSMP]":"",
		      intel_hooked?" [Intel KMPC]":"",
		      gnu_hooked?" [GNU GOMP]":"");
		}

		/*
		* If we hooked any compiler-specific routines, just hook for the
		* common OpenMP routines
		*/

		omp_common_get_hook_points_c(0);
		omp_common_get_hook_points_f(0);
	} else {
		fprintf (stdout, PACKAGE_NAME": Warning! You have loaded an OpenMP tracing library but the application seems not to be linked with an OpenMP runtime. Did you compile with the proper flags? (-fopenmp, -openmp, ...).\n");
	}

#else  /* PIC */
	if (me == 0)
	{
		fprintf (stderr, PACKAGE_NAME": Warning! OpenMP instrumentation requires linking with shared library!\n");
	}

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
