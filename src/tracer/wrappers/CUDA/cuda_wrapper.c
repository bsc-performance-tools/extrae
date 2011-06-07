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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/tracer/wrappers/OMP/omp_wrapper.c $
 | @last_commit: $Date: 2010-10-26 14:58:30 +0200 (dt, 26 oct 2010) $
 | @version:     $Revision: 476 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: omp_wrapper.c 476 2010-10-26 12:58:30Z harald $";

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

/* The three following lines are convenient hacks to avoid including cuda.h */
typedef int cudaError_t;
typedef int cudaMemcpyKind_t;
struct dim3 { unsigned int x, y, z; };

static cudaError_t (*real_cudaLaunch)(char*) = NULL;
static cudaError_t (*real_cudaConfigureCall)(struct dim3 p1, struct dim3 p2, size_t p3, void *p4) = NULL;
static cudaError_t (*real_cudaThreadSynchronize)(void) = NULL;
static cudaError_t (*real_cudaMemcpy)(void*,void*,size_t,cudaMemcpyKind_t) = NULL;

void cuda_tracing_init(int rank)
{
	real_cudaLaunch = (cudaError_t(*)(char*)) dlsym (RTLD_NEXT, "cudaLaunch");
	if (real_cudaLaunch == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaLaunch in DSOs!!\n");

#if 0
	real_cudaConfigureCall = (cudaError_t(*)(struct dim3 p1, struct dim3 p2, size_t p3, void *p4)) dlsym (RTLD_NEXT, "cudaConfigureCall");
	if (real_cudaConfigureCall == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaConfigureCall in DSOs!!\n");
#endif

	real_cudaThreadSynchronize = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaThreadSynchronize");
	if (real_cudaThreadSynchronize == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaThreadSynchronize in DSOs!!\n");

	real_cudaMemcpy = (cudaError_t(*)(void*,void*,size_t,cudaMemcpyKind_t)) dlsym (RTLD_NEXT, "cudaMemcpy");
	if (real_cudaMemcpy == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaMemcpy in DSOs!!\n");
}

cudaError_t cudaLaunch (char *p1)
{
	cudaError_t res;

	if (real_cudaLaunch == NULL)
	{
		fprintf (stderr, "Unable to find cudaLaunch in DSOs! Dying...\n");
		exit (0);
	}

	Backend_Enter_Instrumentation (1);
	Probe_Cuda_Launch_Entry ();
	res = real_cudaLaunch (p1);
	Probe_Cuda_Launch_Exit ();
	Backend_Leave_Instrumentation ();

	return res;
}

#if 0
cudaError_t cudaConfigureCall (struct dim3 p1, struct dim3 p2, size_t p3, void *p4) /* cudaStream_t */
{
	cudaError_t res;

	if (real_cudaConfigureCall == NULL)
	{
		fprintf (stderr, "Unable to find cudaConfigureCall in DSOs!! Dying...\n");
		exit (0);
	}

	res = real_cudaConfigureCall (p1, p2, p3, p4);

	return res;
}
#endif

cudaError_t cudaThreadSynchronize (void)
{
	cudaError_t res;

	if (real_cudaThreadSynchronize == NULL)
	{
		fprintf (stderr, "Unable to find cudaThreadSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	Backend_Enter_Instrumentation (1);
	Probe_Cuda_Barrier_Entry ();
	res = real_cudaThreadSynchronize ();
	Probe_Cuda_Barrier_Exit ();
	Backend_Leave_Instrumentation ();

	return res;
}

cudaError_t cudaMemcpy (void *p1, void *p2 , size_t p3, cudaMemcpyKind_t p4)
{
	cudaError_t res;

	if (real_cudaMemcpy == NULL)
	{
		fprintf (stderr, "Unable to find cudaMemcpy in DSOs!!\n");
		exit (0);
	}

	Backend_Enter_Instrumentation (1);
	Probe_Cuda_Memcpy_Entry (p3);
	res = real_cudaMemcpy (p1, p2, p3, p4);
	Probe_Cuda_Memcpy_Exit ();
	Backend_Leave_Instrumentation ();

	return res;
}
