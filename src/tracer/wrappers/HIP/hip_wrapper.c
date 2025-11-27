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

#include "hip_common.h"
#include "hip_probe.h"
#include "wrapper.h"

//#define DEBUG

/**
 ** Regular LD_PRELOAD instrumentation
 **/
#if defined(PIC)
static hipError_t (*real_hipLaunch)(const void*) = NULL;
static hipError_t (*real_hipConfigureCall)(dim3, dim3, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipThreadSynchronize)(void) = NULL;
static hipError_t (*real_hipDeviceSynchronize)(void) = NULL;
static hipError_t (*real_hipStreamSynchronize)(hipStream_t) = NULL;
static hipError_t (*real_hipMemcpy)(void*,const void*,size_t,enum hipMemcpyKind) = NULL;
static hipError_t (*real_hipMemcpyAsync)(void*,const void*,size_t,enum hipMemcpyKind,hipStream_t) = NULL;
static hipError_t (*real_hipStreamCreate)(hipStream_t*) = NULL;
static hipError_t (*real_hipStreamCreateWithFlags)(hipStream_t*, unsigned int) = NULL;
static hipError_t (*real_hipStreamCreateWithPriority)(hipStream_t*, unsigned int, int) = NULL;
static hipError_t (*real_hipStreamDestroy)(hipStream_t) = NULL;
static hipError_t (*real_hipDeviceReset)(void) = NULL;
static hipError_t (*real_hipThreadExit)(void) = NULL;

static hipError_t (*real_hipMalloc)(void **, size_t) = NULL;
static hipError_t (*real_hipMallocPitch)(void **, size_t *, size_t, size_t) = NULL;
static hipError_t (*real_hipFree)(void *) = NULL;
static hipError_t (*real_hipMallocArray)(hipArray_t *, const struct hipChannelFormatDesc *, size_t, size_t, unsigned int) = NULL;
static hipError_t (*real_hipFreeArray)(hipArray_t) = NULL;
static hipError_t (*real_hipMallocHost)(void **, size_t) = NULL;
static hipError_t (*real_hipFreeHost)(void *) = NULL;
static hipError_t (*real_hipHostAlloc)(void **, size_t, unsigned int) = NULL;
static hipError_t (*real_hipMemset)(void *, int, size_t) = NULL;

/* v6.0 */
static hipError_t (*real_hipMallocManaged)(void **, size_t, unsigned int) = NULL;

/* v7.0 */
static hipError_t (*real_hipLaunchKernel)(const void*, dim3, dim3, void**, size_t, hipStream_t) = NULL;

#endif /* PIC */

void Extrae_HIP_init (int rank)
{
	UNREFERENCED_PARAMETER(rank);

#if defined(PIC)
	real_hipLaunch = (hipError_t(*)(const void*)) dlsym (RTLD_NEXT, "hipLaunch");

	real_hipConfigureCall = (hipError_t(*)(dim3, dim3, size_t, hipStream_t)) dlsym (RTLD_NEXT, "hipConfigureCall");

	real_hipThreadSynchronize = (hipError_t(*)(void)) dlsym (RTLD_NEXT, "hipThreadSynchronize");

	real_hipDeviceSynchronize = (hipError_t(*)(void)) dlsym (RTLD_NEXT, "hipDeviceSynchronize");

	real_hipStreamSynchronize = (hipError_t(*)(hipStream_t)) dlsym (RTLD_NEXT, "hipStreamSynchronize");

	real_hipMemcpy = (hipError_t(*)(void*,const void*,size_t,enum hipMemcpyKind)) dlsym (RTLD_NEXT, "hipMemcpy");

	real_hipMemcpyAsync = (hipError_t(*)(void*,const void*,size_t,enum hipMemcpyKind,hipStream_t)) dlsym (RTLD_NEXT, "hipMemcpyAsync");

	real_hipStreamCreate = (hipError_t(*)(hipStream_t*)) dlsym (RTLD_NEXT, "hipStreamCreate");

	real_hipStreamCreateWithFlags = (hipError_t(*)(hipStream_t*, unsigned int)) dlsym (RTLD_NEXT, "hipStreamCreateWithFlags");

	real_hipStreamCreateWithPriority = (hipError_t(*)(hipStream_t*, unsigned int, int)) dlsym (RTLD_NEXT, "hipStreamCreateWithPriority");

	real_hipStreamDestroy = (hipError_t(*)(hipStream_t)) dlsym (RTLD_NEXT, "hipStreamDestroy");

	real_hipDeviceReset = (hipError_t(*)(void)) dlsym (RTLD_NEXT, "hipDeviceReset");

	real_hipThreadExit = (hipError_t(*)(void)) dlsym (RTLD_NEXT, "hipThreadExit");

	real_hipMalloc = (hipError_t(*)(void **, size_t))dlsym(RTLD_NEXT, "hipMalloc");
	real_hipMallocPitch = (hipError_t(*)(void **, size_t *, size_t, size_t))dlsym(RTLD_NEXT, "hipMallocPitch");
	real_hipFree = (hipError_t(*)(void *))dlsym(RTLD_NEXT, "hipFree");
	real_hipMallocArray = (hipError_t(*)(hipArray_t *, const struct hipChannelFormatDesc *, size_t, size_t, unsigned int))dlsym(RTLD_NEXT, "hipMallocArray");
	real_hipFreeArray = (hipError_t(*)(hipArray_t))dlsym(RTLD_NEXT, "hipFreeArray");
	real_hipMallocHost = (hipError_t(*)(void **, size_t))dlsym(RTLD_NEXT, "hipMallocHost");
	real_hipFreeHost = (hipError_t(*)(void *))dlsym(RTLD_NEXT, "hipFreeHost");
	real_hipHostAlloc = (hipError_t(*)(void **, size_t, unsigned int))dlsym(RTLD_NEXT, "hipHostAlloc");
	real_hipMemset = (hipError_t(*)(void *, int, size_t))dlsym(RTLD_NEXT, "hipMemset");

/* 6.0 */
	real_hipMallocManaged = (hipError_t(*)(void **, size_t, unsigned int))dlsym(RTLD_NEXT, "hipMallocManaged");

/* 7.0 */
	real_hipLaunchKernel = (hipError_t(*)(const void*, dim3, dim3, void**, size_t, hipStream_t)) dlsym(RTLD_NEXT, "hipLaunchKernel");
#else
	fprintf (stderr, PACKAGE_NAME": Warning! HIP instrumentation requires linking with shared library!\n");
#endif /* PIC */
}

/*
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
*/

#if defined(PIC)

#if 0
static int _hipLaunch_device = 0;
static int _hipLaunch_stream = 0;
#endif

hipError_t hipLaunch (const void *func)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipLaunch is at %p\n", THREADID, real_hipLaunch);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipLaunch params %p\n", THREADID, func);
#endif

	if (real_hipLaunch != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipLaunch_Enter (func, NULL);
		res = real_hipLaunch (func);
		Extrae_hipLaunch_Exit ();
	}
	else if (real_hipLaunch != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipLaunch (func);
	}
	else
	{
		fprintf (stderr, "Unable to find hipLaunch in DSOs! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipConfigureCall (dim3 p1, dim3 p2, size_t p3, hipStream_t p4)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipConfigureCall is at %p\n", THREADID, real_hipConfigureCall);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipConfigureCall params p1 p2 %d %d\n", THREADID, p3, p4);
#endif

	if (real_hipConfigureCall != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipConfigureCall_Enter (p1, p2, p3, p4);
		res = real_hipConfigureCall (p1, p2, p3, p4);
		Extrae_hipConfigureCall_Exit ();
	}
	else if (real_hipConfigureCall != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipConfigureCall (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, "Unable to find hipConfigureCall in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipStreamCreate (hipStream_t *pStream)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamCreate is at %p\n", THREADID, real_hipStreamCreate);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamCreate params %p\n", THREADID, pStream);
#endif

	if (real_hipStreamCreate != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipStreamCreate_Enter (pStream);
		res = real_hipStreamCreate (pStream);
		Extrae_hipStreamCreate_Exit ();
	}
	else if (real_hipStreamCreate != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipStreamCreate (pStream);
	}
	else
	{
		fprintf (stderr, "Unable to find hipStreamCreate in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipStreamCreateWithFlags (hipStream_t *pStream, unsigned int flags)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamCreateWithFlags is at %p\n", THREADID, real_hipStreamCreateWithFlags);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamCreateWithFlags params %p %u\n", THREADID, pStream, flags);
#endif

	if (real_hipStreamCreateWithFlags != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipStreamCreate_Enter (pStream);
		res = real_hipStreamCreateWithFlags (pStream, flags);
		Extrae_hipStreamCreate_Exit ();
	}
	else if (real_hipStreamCreateWithFlags != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipStreamCreateWithFlags (pStream, flags);
	}
	else
	{
		fprintf (stderr, "Unable to find hipStreamCreateWithFlags in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipStreamCreateWithPriority (hipStream_t *pStream, unsigned int flags, int priority)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamCreateWithPriority is at %p\n", THREADID, real_hipStreamCreateWithFlags);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamCreateWithPriority params %p %u %d\n", THREADID, pStream, flags, priority);
#endif

	if (real_hipStreamCreateWithPriority != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipStreamCreate_Enter (pStream);
		res = real_hipStreamCreateWithPriority (pStream, flags, priority);
		Extrae_hipStreamCreate_Exit ();
	}
	else if (real_hipStreamCreateWithPriority != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipStreamCreateWithPriority (pStream, flags, priority);
	}
	else
	{
		fprintf (stderr, "Unable to find hipStreamCreateWithPriority in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipStreamDestroy (hipStream_t stream)
{
	hipError_t res;

#if defined (DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamDestroy is at %p\n", THREADID, real_hipStreamDestroy);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamDestroy params %p\n", THREADID, stream);
#endif

	if (real_hipStreamDestroy != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipStreamDestroy_Enter (stream);
		res = real_hipStreamDestroy (stream);
		Extrae_hipStreamDestroy_Exit ();
	}
	else if (real_hipStreamDestroy != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipStreamDestroy (stream);
	}
	else
	{
		fprintf (stderr, "Unable to find hipStreamDestroy in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipMemcpyAsync (void *p1, const void *p2, size_t p3, enum hipMemcpyKind p4, hipStream_t p5)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipMemcpyAsync is at %p\n", THREADID, real_hipMemcpyAsync);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipMemcpyAsync params %p %p %d %d %d\n", THREADID, p1, p2, p3, p4, p5);
#endif

	if (real_hipMemcpyAsync != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMemcpyAsync_Enter (p1, p2, p3, p4, p5);
		res = real_hipMemcpyAsync (p1, p2, p3, p4, p5);
		Extrae_hipMemcpyAsync_Exit ();
	}
	else if (real_hipMemcpyAsync != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMemcpyAsync (p1, p2, p3, p4, p5);
	}
	else
	{
		fprintf (stderr, "Unable to find hipMemcpyAsync in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipMemcpy (void *p1, const void *p2, size_t p3, enum hipMemcpyKind p4)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipMemcpy is at %p\n", THREADID, real_hipMemcpy);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipMemcpy params %p %p %d %d\n", THREADID, p1, p2, p3, p4);
#endif

	if (real_hipMemcpy != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMemcpy_Enter (p1, p2, p3, p4);
		res = real_hipMemcpy (p1, p2, p3, p4);
		Extrae_hipMemcpy_Exit ();
	}
	else if (real_hipMemcpy != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMemcpy (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, "Unable to find hipMemcpy in DSOs!! Dying...\n");
		exit (0);
	}
	return res;
}

hipError_t hipThreadSynchronize (void)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipThreadSynchronize is at %p\n", THREADID, real_hipThreadSynchronize);
#endif

	if (real_hipThreadSynchronize != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipThreadSynchronize_Enter ();
		res = real_hipThreadSynchronize ();
		Extrae_hipThreadSynchronize_Exit ();
	}
	else if (real_hipThreadSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipThreadSynchronize ();
	}
	else
	{
		fprintf (stderr, "Unable to find hipThreadSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipDeviceSynchronize (void)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipDeviceSynchronize is at %p\n", THREADID, real_hipThreadSynchronize);
#endif

	if (real_hipDeviceSynchronize != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipDeviceSynchronize_Enter ();
		res = real_hipDeviceSynchronize ();
		Extrae_hipDeviceSynchronize_Exit ();
	}
	else if (real_hipDeviceSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipDeviceSynchronize ();
	}
	else
	{
		fprintf (stderr, "Unable to find hipDeviceSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipStreamSynchronize (hipStream_t p1)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamSynchronize is at %p\n", THREADID, real_hipStreamSynchronize);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipStreamSynchronize params %d\n", THREADID, p1);
#endif

	if (real_hipStreamSynchronize != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipStreamSynchronize_Enter (p1);
		res = real_hipStreamSynchronize (p1);
		Extrae_hipStreamSynchronize_Exit ();
	}
	else if (real_hipStreamSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipStreamSynchronize (p1);
	}
	else
	{
		fprintf (stderr, "Unable to find hipStreamSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t hipThreadExit (void)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipThreadExit is at %p\n", THREADID, real_hipThreadExit);
#endif

	if (real_hipThreadExit != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipThreadExit_Enter ();
		res = real_hipThreadExit ();
		Extrae_hipThreadExit_Exit ();
	}
	else if (real_hipStreamSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipThreadExit ();
	}
	else
	{
		fprintf (stderr, "Unable to find hipThreadExit in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}


hipError_t hipDeviceReset (void)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipDeviceReset is at %p\n", THREADID, real_hipDeviceReset);
#endif

	if (real_hipDeviceReset != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipDeviceReset_Enter ();
		res = real_hipDeviceReset ();
		Extrae_hipDeviceReset_Exit ();
	}
	else if (real_hipStreamSynchronize != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipDeviceReset ();
	}
	else
	{
		fprintf (stderr, "Unable to find hipDeviceReset in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

hipError_t
hipMalloc(void **devPtr, size_t size)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipMalloc);
#endif

	if (real_hipMalloc != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMalloc_Enter(HIPMALLOC_VAL, devPtr, size);
		res = real_hipMalloc(devPtr, size);
		Extrae_hipMalloc_Exit(HIPMALLOC_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMalloc(devPtr, size);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipMallocPitch);
#endif

	if  (real_hipMallocPitch != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMalloc_Enter(HIPMALLOCPITCH_VAL, devPtr, width * height);
		res = real_hipMallocPitch(devPtr, pitch, width, height);
		Extrae_hipMalloc_Exit(HIPMALLOCPITCH_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMallocPitch(devPtr, pitch, width, height);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipFree(void *devPtr)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipFree);
#endif

	if  (real_hipFree != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipFree_Enter(HIPFREE_VAL, devPtr);
		res = real_hipFree(devPtr);
		Extrae_hipFree_Exit(HIPFREE_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipFree(devPtr);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipMallocArray(hipArray_t *array, const struct hipChannelFormatDesc *desc,
  size_t width, size_t height, unsigned int flags)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipMallocArray);
#endif

	if  (real_hipMallocArray != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMalloc_Enter(
		  HIPMALLOCARRAY_VAL, (void *)array, width * height
		  );
		res = real_hipMallocArray(array, desc, width, height, flags);
		Extrae_hipMalloc_Exit(HIPMALLOCARRAY_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMallocArray(array, desc, width, height, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipFreeArray(hipArray_t array)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipFreeArray);
#endif

	if  (real_hipFreeArray != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipFree_Enter(HIPFREEARRAY_VAL, (void *)array);
		res = real_hipFreeArray(array);
		Extrae_hipFree_Exit(HIPFREEARRAY_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipFreeArray(array);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipMallocHost(void **ptr, size_t size)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipMallocHost);
#endif

	if  (real_hipMallocHost != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMalloc_Enter(HIPMALLOCHOST_VAL, ptr, size);
		res = real_hipMallocHost(ptr, size);
		Extrae_hipMalloc_Exit(HIPMALLOCHOST_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMallocHost(ptr, size);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipFreeHost(void *ptr)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipFreeHost);
#endif

	if  (real_hipFreeHost != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipFree_Enter(HIPFREEHOST_VAL, ptr);
		res = real_hipFreeHost(ptr);
		Extrae_hipFree_Exit(HIPFREEHOST_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipFreeHost(ptr);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipHostAlloc(void **pHost, size_t size, unsigned int flags)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipHostAlloc);
#endif

	if  (real_hipHostAlloc != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipHostAlloc_Enter(pHost, size);
		res = real_hipHostAlloc(pHost, size, flags);
		Extrae_hipHostAlloc_Exit();
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipHostAlloc(pHost, size, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

hipError_t
hipMemset(void *devPtr, int value, size_t count)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipMemset);
#endif

	if  (real_hipMemset != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMemset_Enter(devPtr, count);
		res = real_hipMemset(devPtr, value, count);
		Extrae_hipMemset_Exit();
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMemset(devPtr, value, count);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

/* v6.0 */
hipError_t
hipMallocManaged(void** devPtr, size_t size, unsigned int flags)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf(stderr,
	  PACKAGE_NAME": THREAD %d %s is at %p\n",
	  THREADID, __func__, real_hipMallocManaged);
#endif

	if (real_hipMallocManaged != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipMalloc_Enter(HIPMALLOC_VAL, devPtr, size);
		res = real_hipMallocManaged(devPtr, size, flags);
		Extrae_hipMalloc_Exit(HIPMALLOC_VAL);
	}
	else if (real_hipStreamSynchronize != NULL &&
	  !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipMallocManaged(devPtr, size, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying ...\n", __func__);
		exit(0);
	}

	return res;
}

/* v7.0 */
hipError_t
hipLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, hipStream_t stream)
{
	hipError_t res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipLaunchKernel is at %p\n", THREADID, real_hipLaunchKernel);
	fprintf (stderr, PACKAGE_NAME": THREAD %d hipLaunchKernel params %p %p %p %p %p %p\n", THREADID, func, gridDim, blockDim, args, sharedMem, stream);
#endif

	if (real_hipLaunchKernel != NULL && mpitrace_on && Extrae_get_trace_HIP())
	{
		Extrae_hipLaunch_Enter(func, stream);
		res = real_hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
		Extrae_hipLaunch_Exit();
	}
	else if (real_hipLaunchKernel != NULL && !(mpitrace_on && Extrae_get_trace_HIP()))
	{
		res = real_hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
	}
	else
	{
		fprintf (stderr, "Unable to find hipLaunchKernel in DSOs! Dying...\n");
		exit (0);
	}

	return res;
}

#endif /* PIC */
