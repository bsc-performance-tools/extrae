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

#include "cuda_common.h"
#include "cuda_probe.h"
#include "wrapper.h"

//#define DEBUG

/**
 ** Regular LD_PRELOAD instrumentation
 **/
#if defined(PIC)
static cudaError_t (*real_cudaLaunch)(const char*) = NULL;
static cudaError_t (*real_cudaConfigureCall)(dim3, dim3, size_t, cudaStream_t) = NULL;
static cudaError_t (*real_cudaThreadSynchronize)(void) = NULL;
static cudaError_t (*real_cudaDeviceSynchronize)(void) = NULL;
static cudaError_t (*real_cudaStreamSynchronize)(cudaStream_t) = NULL;
static cudaError_t (*real_cudaMemcpy)(void*,const void*,size_t,enum cudaMemcpyKind) = NULL;
static cudaError_t (*real_cudaMemcpyAsync)(void*,const void*,size_t,enum cudaMemcpyKind,cudaStream_t) = NULL;
static cudaError_t (*real_cudaStreamCreate)(cudaStream_t*) = NULL;
static cudaError_t (*real_cudaStreamCreateWithFlags)(cudaStream_t*, unsigned int) = NULL;
static cudaError_t (*real_cudaStreamCreateWithPriority)(cudaStream_t*, unsigned int, int) = NULL;
static cudaError_t (*real_cudaStreamDestroy)(cudaStream_t) = NULL;
static cudaError_t (*real_cudaDeviceReset)(void) = NULL;
static cudaError_t (*real_cudaThreadExit)(void) = NULL;

static cudaError_t (*real_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*real_cudaMallocPitch)(void **, size_t *, size_t, size_t) = NULL;
static cudaError_t (*real_cudaFree)(void *) = NULL;
static cudaError_t (*real_cudaMallocArray)(cudaArray_t *, const cudaChannelFormatDesc *, size_t, size_t, unsigned int) = NULL;
static cudaError_t (*real_cudaFreeArray)(cudaArray_t) = NULL;
static cudaError_t (*real_cudaMallocHost)(void **, size_t) = NULL;
static cudaError_t (*real_cudaFreeHost)(void *) = NULL;
static cudaError_t (*real_cudaHostAlloc)(void **, size_t, unsigned int) = NULL;
static cudaError_t (*real_cudaMemset)(void *, int, size_t) = NULL;
#endif /* PIC */

void Extrae_CUDA_init (int rank)
{
	UNREFERENCED_PARAMETER(rank);

#if defined(PIC)
	real_cudaLaunch = (cudaError_t(*)(const char*)) dlsym (RTLD_NEXT, "cudaLaunch");

	real_cudaConfigureCall = (cudaError_t(*)(dim3, dim3, size_t, cudaStream_t)) dlsym (RTLD_NEXT, "cudaConfigureCall");

	real_cudaThreadSynchronize = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaThreadSynchronize");

	real_cudaDeviceSynchronize = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaDeviceSynchronize");

	real_cudaStreamSynchronize = (cudaError_t(*)(cudaStream_t)) dlsym (RTLD_NEXT, "cudaStreamSynchronize");

	real_cudaMemcpy = (cudaError_t(*)(void*,const void*,size_t,enum cudaMemcpyKind)) dlsym (RTLD_NEXT, "cudaMemcpy");

	real_cudaMemcpyAsync = (cudaError_t(*)(void*,const void*,size_t,enum cudaMemcpyKind,cudaStream_t)) dlsym (RTLD_NEXT, "cudaMemcpyAsync");

	real_cudaStreamCreate = (cudaError_t(*)(cudaStream_t*)) dlsym (RTLD_NEXT, "cudaStreamCreate");

	real_cudaStreamCreateWithFlags = (cudaError_t(*)(cudaStream_t*, unsigned int)) dlsym (RTLD_NEXT, "cudaStreamCreateWithFlags");

	real_cudaStreamCreateWithPriority = (cudaError_t(*)(cudaStream_t*, unsigned int, int)) dlsym (RTLD_NEXT, "cudaStreamCreateWithPriority");

	real_cudaStreamDestroy = (cudaError_t(*)(cudaStream_t)) dlsym (RTLD_NEXT, "cudaStreamDestroy");

	real_cudaDeviceReset = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaDeviceReset");

	real_cudaThreadExit = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaThreadExit");

	real_cudaMalloc = (cudaError_t(*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMalloc");
	real_cudaMallocPitch = (cudaError_t(*)(void **, size_t *, size_t, size_t))dlsym(RTLD_NEXT, "cudaMallocPitch");
	real_cudaFree = (cudaError_t(*)(void *))dlsym(RTLD_NEXT, "cudaFree");
	real_cudaMallocArray = (cudaError_t(*)(cudaArray_t *, const cudaChannelFormatDesc *, size_t, size_t, unsigned int))dlsym(RTLD_NEXT, "cudaMallocArray");
	real_cudaFreeArray = (cudaError_t(*)(cudaArray_t))dlsym(RTLD_NEXT, "cudaFreeArray");
	real_cudaMallocHost = (cudaError_t(*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMallocHost");
	real_cudaFreeHost = (cudaError_t(*)(void *))dlsym(RTLD_NEXT, "cudaFreeHost");
	real_cudaHostAlloc = (cudaError_t(*)(void **, size_t, unsigned int))dlsym(RTLD_NEXT, "cudaHostAlloc");
	real_cudaMemset = (cudaError_t(*)(void *, int, size_t))dlsym(RTLD_NEXT, "cudaMemset");
#else
	fprintf (stderr, PACKAGE_NAME": Warning! CUDA instrumentation requires linking with shared library!\n");
#endif /* PIC */
}

/*
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
*/

#if defined(PIC)

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

	if (real_cudaLaunch != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaLaunch_Enter (p1);
		res = real_cudaLaunch (p1);
		Extrae_cudaLaunch_Exit ();
	}
	else if (real_cudaLaunch != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
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

	if (real_cudaConfigureCall != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaConfigureCall_Enter (p1, p2, p3, p4);
		res = real_cudaConfigureCall (p1, p2, p3, p4);
		Extrae_cudaConfigureCall_Exit ();
	}
	else if (real_cudaConfigureCall != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
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

cudaError_t cudaStreamCreate (cudaStream_t *pStream)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreate is at %p\n", THREADID, real_cudaStreamCreate);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreate params %p\n", THREADID, pStream);
#endif

	if (real_cudaStreamCreate != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaStreamCreate_Enter (pStream);
		res = real_cudaStreamCreate (pStream);
		Extrae_cudaStreamCreate_Exit ();
	}
	else if (real_cudaStreamCreate != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaStreamCreate (pStream);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamCreate in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaStreamCreateWithFlags (cudaStream_t *pStream, unsigned int flags)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreateWithFlags is at %p\n", THREADID, real_cudaStreamCreateWithFlags);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreateWithFlags params %p %u\n", THREADID, pStream, flags);
#endif

	if (real_cudaStreamCreateWithFlags != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaStreamCreate_Enter (pStream);
		res = real_cudaStreamCreateWithFlags (pStream, flags);
		Extrae_cudaStreamCreate_Exit ();
	}
	else if (real_cudaStreamCreateWithFlags != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaStreamCreateWithFlags (pStream, flags);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamCreateWithFlags in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaStreamCreateWithPriority (cudaStream_t *pStream, unsigned int flags, int priority)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreateWithPriority is at %p\n", THREADID, real_cudaStreamCreateWithFlags);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamCreateWithPriority params %p %u %d\n", THREADID, pStream, flags, priority);
#endif

	if (real_cudaStreamCreateWithPriority != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaStreamCreate_Enter (pStream);
		res = real_cudaStreamCreateWithPriority (pStream, flags, priority);
		Extrae_cudaStreamCreate_Exit ();
	}
	else if (real_cudaStreamCreateWithPriority != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaStreamCreateWithPriority (pStream, flags, priority);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamCreateWithPriority in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaStreamDestroy (cudaStream_t stream)
{
	cudaError_t res;

#if defined (DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamDestroy is at %p\n", THREADID, real_cudaStreamDestroy);
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaStreamDestroy params %p\n", THREADID, stream);
#endif

	if (real_cudaStreamDestroy != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaStreamDestroy_Enter (stream);
		res = real_cudaStreamDestroy (stream);
		Extrae_cudaStreamDestroy_Exit ();
	}
	else if (real_cudaStreamDestroy != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaStreamDestroy (stream);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamDestroy in DSOs!! Dying...\n");
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

	if (real_cudaMemcpyAsync != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaMemcpyAsync_Enter (p1, p2, p3, p4, p5);
		res = real_cudaMemcpyAsync (p1, p2, p3, p4, p5);
		Extrae_cudaMemcpyAsync_Exit ();
	}
	else if (real_cudaMemcpyAsync != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
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

	if (real_cudaMemcpy != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaMemcpy_Enter (p1, p2, p3, p4);
		res = real_cudaMemcpy (p1, p2, p3, p4);
		Extrae_cudaMemcpy_Exit ();
	}
	else if (real_cudaMemcpy != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
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
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaThreadSynchronize is at %p\n", THREADID, real_cudaThreadSynchronize);
#endif

	if (real_cudaThreadSynchronize != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaThreadSynchronize_Enter ();
		res = real_cudaThreadSynchronize ();
		Extrae_cudaThreadSynchronize_Exit ();
	}
	else if (real_cudaThreadSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
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

cudaError_t cudaDeviceSynchronize (void)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaDeviceSynchronize is at %p\n", THREADID, real_cudaThreadSynchronize);
#endif

	if (real_cudaDeviceSynchronize != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaDeviceSynchronize_Enter ();
		res = real_cudaDeviceSynchronize ();
		Extrae_cudaDeviceSynchronize_Exit ();
	}
	else if (real_cudaDeviceSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaDeviceSynchronize ();
	}
	else
	{
		fprintf (stderr, "Unable to find cudaDeviceSynchronize in DSOs!! Dying...\n");
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

	if (real_cudaStreamSynchronize != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaStreamSynchronize_Enter (p1);
		res = real_cudaStreamSynchronize (p1);
		Extrae_cudaStreamSynchronize_Exit ();
	}
	else if (real_cudaStreamSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
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

cudaError_t cudaThreadExit (void)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaThreadExit is at %p\n", THREADID, real_cudaThreadExit);
#endif

	if (real_cudaThreadExit != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaThreadExit_Enter ();
		res = real_cudaThreadExit ();
		Extrae_cudaThreadExit_Exit ();
	}
	else if (real_cudaStreamSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaThreadExit ();
	}
	else
	{
		fprintf (stderr, "Unable to find cudaThreadExit in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}


cudaError_t cudaDeviceReset (void)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d cudaDeviceReset is at %p\n", THREADID, real_cudaDeviceReset);
#endif

	if (real_cudaDeviceReset != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaDeviceReset_Enter ();
		res = real_cudaDeviceReset ();
		Extrae_cudaDeviceReset_Exit ();
	}
	else if (real_cudaStreamSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_CUDA()))
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

cudaError_t
cudaMalloc(void **devPtr, size_t size)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaMalloc);
#endif

	if (real_cudaMalloc != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaMalloc_Enter(CUDAMALLOC_EV, devPtr, size);
		res = real_cudaMalloc(devPtr, size);
		Extrae_cudaMalloc_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaMalloc(devPtr, size);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaMallocPitch);
#endif

	if  (real_cudaMallocPitch != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaMalloc_Enter(CUDAMALLOCPITCH_EV, devPtr, width * height);
		res = real_cudaMallocPitch(devPtr, pitch, width, height);
		Extrae_cudaMalloc_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaMallocPitch(devPtr, pitch, width, height);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaFree(void *devPtr)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaFree);
#endif

	if  (real_cudaFree != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaFree_Enter(CUDAFREE_EV, devPtr);
		res = real_cudaFree(devPtr);
		Extrae_cudaFree_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaFree(devPtr);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaMallocArray(cudaArray_t *array, const cudaChannelFormatDesc *desc,
  size_t width, size_t height, unsigned int flags)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaMallocArray);
#endif

	if  (real_cudaMallocArray != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaMalloc_Enter(
		  CUDAMALLOCARRAY_EV, (void *)array, width * height
		  );
		res = real_cudaMallocArray(array, desc, width, height, flags);
		Extrae_cudaMalloc_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaMallocArray(array, desc, width, height, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaFreeArray(cudaArray_t array)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaFreeArray);
#endif

	if  (real_cudaFreeArray != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaFree_Enter(CUDAFREEARRAY_EV, (void *)array);
		res = real_cudaFreeArray(array);
		Extrae_cudaFree_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaFreeArray(array);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaMallocHost(void **ptr, size_t size)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaMallocHost);
#endif

	if  (real_cudaMallocHost != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaMalloc_Enter(CUDAMALLOCHOST_EV, ptr, size);
		res = real_cudaMallocHost(ptr, size);
		Extrae_cudaMalloc_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaMallocHost(ptr, size);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaFreeHost(void *ptr)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaFreeHost);
#endif

	if  (real_cudaFreeHost != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaFree_Enter(CUDAFREEHOST_EV, ptr);
		res = real_cudaFreeHost(ptr);
		Extrae_cudaFree_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaFreeHost(ptr);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaHostAlloc);
#endif

	if  (real_cudaHostAlloc != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaHostAlloc_Enter(pHost, size);
		res = real_cudaHostAlloc(pHost, size, flags);
		Extrae_cudaHostAlloc_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaHostAlloc(pHost, size, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

cudaError_t
cudaMemset(void *devPtr, int value, size_t count)
{
	cudaError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_cudaMemset);
#endif

	if  (real_cudaMemset != NULL && mpitrace_on && Extrae_get_trace_CUDA())
	{
		Extrae_cudaMemset_Enter(devPtr, count);
		res = real_cudaMemset(devPtr, value, count);
		Extrae_cudaMemset_Exit();
	}
	else if (real_cudaStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_CUDA()))
	{
		res = real_cudaMemset(devPtr, value, count);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}
#endif /* PIC */
