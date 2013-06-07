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

#include "cuda_common.h"
#include "wrapper.h"

//#define DEBUG

/**
 ** Regular LD_PRELOAD instrumentation
 **/
static cudaError_t (*real_cudaLaunch)(const char*) = NULL;
static cudaError_t (*real_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
static cudaError_t (*real_cudaThreadSynchronize)(void) = NULL;
static cudaError_t (*real_cudaStreamSynchronize)(cudaStream_t) = NULL;
static cudaError_t (*real_cudaMemcpy)(void*,const void*,size_t,enum cudaMemcpyKind) = NULL;
static cudaError_t (*real_cudaMemcpyAsync)(void*,const void*,size_t,enum cudaMemcpyKind,cudaStream_t) = NULL;
static cudaError_t (*real_cudaStreamCreate)(cudaStream_t*) = NULL;
static cudaError_t (*real_cudaDeviceReset)(void) = NULL;

void Extrae_CUDA_init (int rank)
{
	real_cudaLaunch = (cudaError_t(*)(const char*)) dlsym (RTLD_NEXT, "cudaLaunch");
	if (real_cudaLaunch == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaLaunch in DSOs!!\n");

	real_cudaConfigureCall = (cudaError_t(*)(dim3, dim3, size_t, cudaStream_t)) dlsym (RTLD_NEXT, "cudaConfigureCall");
	if (real_cudaConfigureCall == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaConfigureCall in DSOs!!\n");

	real_cudaThreadSynchronize = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaThreadSynchronize");
	if (real_cudaThreadSynchronize == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaThreadSynchronize in DSOs!!\n");

	real_cudaStreamSynchronize = (cudaError_t(*)(cudaStream_t)) dlsym (RTLD_NEXT, "cudaStreamSynchronize");
	if (real_cudaStreamSynchronize == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaStreamSynchronize in DSOs!!\n");

	real_cudaMemcpy = (cudaError_t(*)(void*,const void*,size_t,enum cudaMemcpyKind)) dlsym (RTLD_NEXT, "cudaMemcpy");
	if (real_cudaMemcpy == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaMemcpy in DSOs!!\n");

	real_cudaMemcpyAsync = (cudaError_t(*)(void*,const void*,size_t,enum cudaMemcpyKind,cudaStream_t)) dlsym (RTLD_NEXT, "cudaMemcpyAsync");
	if (real_cudaMemcpyAsync == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaMemcpyAsync in DSOs!!\n");

	real_cudaStreamCreate = (cudaError_t(*)(cudaStream_t*)) dlsym (RTLD_NEXT, "cudaStreamCreate");
	if (real_cudaStreamCreate == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaStreamCreate in DSOs!!\n");

	real_cudaDeviceReset = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaDeviceReset");
	if (real_cudaStreamCreate == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaDeviceReset in DSOs!!\n");
}

void Extrae_CUDA_fini (void)
{
	Extrae_CUDA_flush_all_streams();
}

#if 0
static int _cudaLaunch_device = 0;
static int _cudaLaunch_stream = 0;
#endif

cudaError_t cudaLaunch (const char *p1)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaLaunch is at %p\n", THREADID, real_cudaLaunch);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaLaunch params %p\n", THREADID, p1);
#endif

	if (real_cudaLaunch != NULL && mpitrace_on)
	{
		Extrae_cudaLaunch_Enter (p1);
		res = real_cudaLaunch (p1);
		Extrae_cudaLaunch_Exit ();
	}
	else if (real_cudaLaunch != NULL && !mpitrace_on)
	{
		res = real_cudaLaunch (p1);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaLaunch in DSOs! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaConfigureCall (dim3 p1, dim3 p2, size_t p3, cudaStream_t p4)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaConfigureCall is at %p\n", THREADID, real_cudaConfigureCall);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaConfigureCall params p1 p2 %d %d\n", THREADID, p3, p4);
#endif

	if (real_cudaConfigureCall != NULL && mpitrace_on)
	{
		Extrae_cudaConfigureCall_Enter (p1, p2, p3, p4);
		res = real_cudaConfigureCall (p1, p2, p3, p4);
		Extrae_cudaConfigureCall_Exit ();
	}
	else if (real_cudaConfigureCall != NULL && !mpitrace_on)
	{
		res = real_cudaConfigureCall (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaConfigureCall in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaStreamCreate (cudaStream_t *p1)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreate is at %p\n", THREADID, real_cudaStreamCreate);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreate params %p %p %d %d %d\n", THREADID, p1);
#endif

	if (real_cudaStreamCreate != NULL && mpitrace_on)
	{
		Extrae_cudaStreamCreate_Enter (p1);
		res = real_cudaStreamCreate (p1);
		Extrae_cudaStreamCreate_Exit ();
	}
	else if (real_cudaStreamCreate != NULL && !mpitrace_on)
	{
		res = real_cudaStreamCreate (p1);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamCreate in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaMemcpyAsync (void *p1, const void *p2, size_t p3, enum cudaMemcpyKind p4, cudaStream_t p5)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaMemcpyAsync is at %p\n", THREADID, real_cudaMemcpyAsync);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaMemcpyAsync params %p %p %d %d %d\n", THREADID, p1, p2, p3, p4, p5);
#endif

	if (real_cudaMemcpyAsync != NULL && mpitrace_on)
	{
		Extrae_cudaMemcpyAsync_Enter (p1, p2, p3, p4, p5);
		res = real_cudaMemcpyAsync (p1, p2, p3, p4, p5);
		Extrae_cudaMemcpyAsync_Exit ();
	}
	else if (real_cudaMemcpyAsync != NULL && !mpitrace_on)
	{
		res = real_cudaMemcpyAsync (p1, p2, p3, p4, p5);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaMemcpyAsync in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaMemcpy (void *p1, const void *p2, size_t p3, enum cudaMemcpyKind p4)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaMemcpy is at %p\n", THREADID, real_cudaMemcpy);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaMemcpy params %p %p %d %d\n", THREADID, p1, p2, p3, p4);
#endif

	if (real_cudaMemcpy != NULL && mpitrace_on)
	{
		Extrae_cudaMemcpy_Enter (p1, p2, p3, p4);
		res = real_cudaMemcpy (p1, p2, p3, p4);
		Extrae_cudaMemcpy_Exit ();
	}
	else if (real_cudaMemcpy != NULL && !mpitrace_on)
	{
		res = real_cudaMemcpy (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaMemcpy in DSOs!! Dying...\n");
		exit (0);
	}
	return res;
}

cudaError_t cudaThreadSynchronize (void)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamSynchronize is at %p\n", THREADID, real_cudaThreadSynchronize);
#endif

	if (real_cudaThreadSynchronize != NULL && mpitrace_on)
	{
		Extrae_cudaThreadSynchronize_Enter ();
		res = real_cudaThreadSynchronize ();
		Extrae_cudaThreadSynchronize_Exit ();
	}
	else if (real_cudaThreadSynchronize != NULL && !mpitrace_on)
	{
		res = real_cudaThreadSynchronize ();
	}
	else
	{
		fprintf (stderr, "Unable to find cudaThreadSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaStreamSynchronize (cudaStream_t p1)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamSynchronize is at %p\n", THREADID, real_cudaStreamSynchronize);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamSynchronize params %d\n", THREADID, p1);
#endif

	if (real_cudaStreamSynchronize != NULL && mpitrace_on)
	{
		Extrae_cudaStreamSynchronize_Enter (p1);
		res = real_cudaStreamSynchronize (p1);
		Extrae_cudaStreamSynchronize_Exit ();
	}
	else if (real_cudaStreamSynchronize != NULL && !mpitrace_on)
	{
		res = real_cudaStreamSynchronize (p1);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaDeviceReset (void)
{
	cudaError_t res;

	if (real_cudaDeviceReset != NULL && mpitrace_on)
	{
		int devid;
		res = real_cudaDeviceReset ();
		cudaGetDevice (&devid);
		Extrae_CUDA_deInitialize (devid);
	}
	else if (real_cudaStreamSynchronize != NULL && !mpitrace_on)
	{
		res = real_cudaDeviceReset ();
	}
	else
	{
		fprintf (stderr, "Unable to find cudaDeviceReset in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

