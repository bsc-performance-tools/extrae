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
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#if !defined(DYNINST_MODULE)

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

#include "wrapper.h"
#include "trace_macros.h"
#include "omp_probe.h"

#include "ibm-xlsmp-1.6.h"
#include "gnu-libgomp-4.2.h"

static void (*omp_set_lock_real)(int *) = NULL;
static void (*omp_unset_lock_real)(int *) = NULL;
static void (*omp_set_num_threads_real)(int) = NULL;
static int (*omp_get_num_threads_real)(void) = NULL;

static void common_GetOpenMPHookPoints (int rank)
{
	/* Obtain @ for omp_set_lock */
	omp_set_lock_real =
		(void(*)(int*)) dlsym (RTLD_NEXT, "omp_set_lock");
	if (omp_set_lock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find omp_set_lock in DSOs!!\n");

	/* Obtain @ for omp_unset_lock */
	omp_unset_lock_real =
		(void(*)(int*)) dlsym (RTLD_NEXT, "omp_unset_lock");
	if (omp_unset_lock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find omp_unset_lock in DSOs!!\n");

	/* Obtain @ for omp_set_num_threads */
	omp_set_num_threads_real =
		(void(*)(int)) dlsym (RTLD_NEXT, "omp_set_num_threads");
	if (omp_set_num_threads_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find omp_set_num_threads in DSOs!!\n");

	/* Obtain @ for omp_get_num_threads */
	omp_get_num_threads_real =
		(int(*)(void)) dlsym (RTLD_NEXT, "omp_get_num_threads");
	if (omp_get_num_threads_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find omp_get_num_threads in DSOs!!\n");
}

/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/

void omp_set_lock (int *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": omp_set_lock is at %p\n", omp_set_lock);
	fprintf (stderr, PACKAGE_NAME": omp_set_lock params %p\n", p1);
#endif

	if (omp_set_lock_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Lock_Entry();
		omp_set_lock_real(p1);
		Probe_OpenMP_Named_Lock_Exit();
		Backend_Leave_Instrumentation ();
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

	if (omp_unset_lock_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Lock_Entry();
		omp_unset_lock_real (p1);
		Probe_OpenMP_Named_Lock_Exit();
		Backend_Leave_Instrumentation ();
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

	if (omp_set_num_threads_real != NULL)
	{
		Backend_ChangeNumberOfThreads (p1);

		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_SetNumThreads_Entry (p1);
		omp_set_num_threads_real (p1);
		Probe_OpenMP_SetNumThreads_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": omp_set_num_threads is not hooked! exiting!!\n");
		exit (0);
	}
}

int omp_get_num_threads (int p1)
{
	int res;

#if defined(DEBUG)
  fprintf (stderr, PACKAGE_NAME": omp_get_num_threads is at %p\n", omp_get_num_threads_real);
  fprintf (stderr, PACKAGE_NAME": omp_get_num_threads params %d\n", p1);
#endif

  if (omp_get_num_threads_real != NULL)
  {
    Backend_Enter_Instrumentation (2);
		Probe_OpenMP_GetNumThreads_Entry ();
    res = omp_get_num_threads_real ();
		Probe_OpenMP_GetNumThreads_Exit ();
    Backend_Leave_Instrumentation ();
  }
  else
  {
    fprintf (stderr, PACKAGE_NAME": omp_set_num_threads is not hooked! exiting!!\n");
    exit (0);
  }

	return res;
}


extern int omp_get_max_threads();

/*
  This __attribute__ tells the loader to run this routine when
  the shared library is loaded 
*/
/* 
	This is IGNORED in SARA ¿? Moved to Backend_preInitialize
  void __attribute__ ((constructor)) openmp_tracing_init(void);
*/
void openmp_tracing_init (void)
{
	int hooked;

#if defined(OS_LINUX) && defined(ARCH_PPC)
	/* On PPC systems, check first for IBM XL runtime, if we don't find any
	   symbol, check for GNU then */
	hooked = ibm_xlsmp_1_6_hook_points(0);
	if (!hooked)
	{
		fprintf (stdout, PACKAGE_NAME": ATTENTION! Application seems not to be linked with IBM XL OpenMP runtime!\n");
		hooked = gnu_libgomp_4_2_hook_points(0);
	}
#else
	hooked = gnu_libgomp_4_2_hook_points(0);
#endif

	if (!hooked)
		fprintf (stdout, PACKAGE_NAME": ATTENTION! Application seems not to be linked with GNU OpenMP runtime!\n");

	/* If we hooked any compiler-specific routines, just hook for the 
	   common OpenMP routines */
	if (hooked)
		common_GetOpenMPHookPoints(0);
}

#endif /* !defined(DYNINST_MODULE) */
