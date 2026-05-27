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

#include "gpu_macros.h"
#include "gpu_common.h"
#include "gpu_probe.h"
#include "wrapper.h"

//#define DEBUG

/* ══════════════════════════════════════════════════════════════════════════
 * Function pointers — resolved at runtime via dlsym
 * ══════════════════════════════════════════════════════════════════════════ */
#if defined(PIC)
static GPU_ERROR_T (*REAL_GPU(Launch))(const void*) = NULL;
static GPU_ERROR_T (*REAL_GPU(ConfigureCall))(dim3, dim3, size_t, GPU_STREAM_T) = NULL;
static GPU_ERROR_T (*REAL_GPU(ThreadSynchronize))(void) = NULL;
static GPU_ERROR_T (*REAL_GPU(DeviceSynchronize))(void) = NULL;
static GPU_ERROR_T (*REAL_GPU(StreamSynchronize))(GPU_STREAM_T) = NULL;
static GPU_ERROR_T (*REAL_GPU(Memcpy))(void*, const void*, size_t, GPU_MEMCPY_KIND_T) = NULL;
static GPU_ERROR_T (*REAL_GPU(MemcpyAsync))(void*, const void*, size_t, GPU_MEMCPY_KIND_T, GPU_STREAM_T) = NULL;
static GPU_ERROR_T (*REAL_GPU(StreamCreate))(GPU_STREAM_T*) = NULL;
static GPU_ERROR_T (*REAL_GPU(StreamCreateWithFlags))(GPU_STREAM_T*, unsigned int) = NULL;
static GPU_ERROR_T (*REAL_GPU(StreamCreateWithPriority))(GPU_STREAM_T*, unsigned int, int) = NULL;
static GPU_ERROR_T (*REAL_GPU(StreamDestroy))(GPU_STREAM_T) = NULL;
static GPU_ERROR_T (*REAL_GPU(DeviceReset))(void) = NULL;
static GPU_ERROR_T (*REAL_GPU(ThreadExit))(void) = NULL;
static GPU_ERROR_T (*REAL_GPU(Malloc))(void**, size_t) = NULL;
static GPU_ERROR_T (*REAL_GPU(MallocPitch))(void**, size_t*, size_t, size_t) = NULL;
static GPU_ERROR_T (*REAL_GPU(Free))(void*) = NULL;
static GPU_ERROR_T (*REAL_GPU(MallocArray))(GPU_ARRAY_T*, const struct GPU_CHANNEL_FORMAT_DESC*, size_t, size_t, unsigned int) = NULL;
static GPU_ERROR_T (*REAL_GPU(FreeArray))(GPU_ARRAY_T) = NULL;
static GPU_ERROR_T (*REAL_GPU(MallocHost))(void**, size_t) = NULL;
static GPU_ERROR_T (*REAL_GPU(FreeHost))(void*) = NULL;
static GPU_ERROR_T (*REAL_GPU(HostAlloc))(void**, size_t, unsigned int) = NULL;
static GPU_ERROR_T (*REAL_GPU(Memset))(void*, int, size_t) = NULL;
/* v6.0 */
static GPU_ERROR_T (*REAL_GPU(MallocManaged))(void**, size_t, unsigned int) = NULL;
/* v7.0 */
static GPU_ERROR_T (*REAL_GPU(LaunchKernel))(const void*, dim3, dim3, void**, size_t, GPU_STREAM_T) = NULL;
#endif /* PIC */

/* ══════════════════════════════════════════════════════════════════════════
 * Extrae_CUDA_init / Extrae_HIP_init — resolve all function pointers via dlsym
 * ══════════════════════════════════════════════════════════════════════════ */
#if defined(CUDA_SUPPORT)
void Extrae_CUDA_init(int rank)
#elif defined(HIP_SUPPORT)
void Extrae_HIP_init(int rank)
#endif
{
	UNREFERENCED_PARAMETER(rank);
	fprintf(stderr, PACKAGE_NAME": [GPU-BACKEND] Using legacy LD_PRELOAD wrapper (gpu_wrapper)\n");

#if defined(PIC)
	REAL_GPU(Launch)                   = (GPU_ERROR_T(*)(const void*))                                                    dlsym(RTLD_NEXT, GPU_SYM(Launch));
	REAL_GPU(ConfigureCall)            = (GPU_ERROR_T(*)(dim3, dim3, size_t, GPU_STREAM_T))                               dlsym(RTLD_NEXT, GPU_SYM(ConfigureCall));
	REAL_GPU(ThreadSynchronize)        = (GPU_ERROR_T(*)(void))                                                           dlsym(RTLD_NEXT, GPU_SYM(ThreadSynchronize));
	REAL_GPU(DeviceSynchronize)        = (GPU_ERROR_T(*)(void))                                                           dlsym(RTLD_NEXT, GPU_SYM(DeviceSynchronize));
	REAL_GPU(StreamSynchronize)        = (GPU_ERROR_T(*)(GPU_STREAM_T))                                                   dlsym(RTLD_NEXT, GPU_SYM(StreamSynchronize));
	REAL_GPU(Memcpy)                   = (GPU_ERROR_T(*)(void*, const void*, size_t, GPU_MEMCPY_KIND_T))                  dlsym(RTLD_NEXT, GPU_SYM(Memcpy));
	REAL_GPU(MemcpyAsync)              = (GPU_ERROR_T(*)(void*, const void*, size_t, GPU_MEMCPY_KIND_T, GPU_STREAM_T))    dlsym(RTLD_NEXT, GPU_SYM(MemcpyAsync));
	REAL_GPU(StreamCreate)             = (GPU_ERROR_T(*)(GPU_STREAM_T*))                                                  dlsym(RTLD_NEXT, GPU_SYM(StreamCreate));
	REAL_GPU(StreamCreateWithFlags)    = (GPU_ERROR_T(*)(GPU_STREAM_T*, unsigned int))                                    dlsym(RTLD_NEXT, GPU_SYM(StreamCreateWithFlags));
	REAL_GPU(StreamCreateWithPriority) = (GPU_ERROR_T(*)(GPU_STREAM_T*, unsigned int, int))                               dlsym(RTLD_NEXT, GPU_SYM(StreamCreateWithPriority));
	REAL_GPU(StreamDestroy)            = (GPU_ERROR_T(*)(GPU_STREAM_T))                                                   dlsym(RTLD_NEXT, GPU_SYM(StreamDestroy));
	REAL_GPU(DeviceReset)              = (GPU_ERROR_T(*)(void))                                                           dlsym(RTLD_NEXT, GPU_SYM(DeviceReset));
	REAL_GPU(ThreadExit)               = (GPU_ERROR_T(*)(void))                                                           dlsym(RTLD_NEXT, GPU_SYM(ThreadExit));
	REAL_GPU(Malloc)                   = (GPU_ERROR_T(*)(void**, size_t))                                                 dlsym(RTLD_NEXT, GPU_SYM(Malloc));
	REAL_GPU(MallocPitch)              = (GPU_ERROR_T(*)(void**, size_t*, size_t, size_t))                                dlsym(RTLD_NEXT, GPU_SYM(MallocPitch));
	REAL_GPU(Free)                     = (GPU_ERROR_T(*)(void*))                                                          dlsym(RTLD_NEXT, GPU_SYM(Free));
	REAL_GPU(MallocArray)              = (GPU_ERROR_T(*)(GPU_ARRAY_T*, const struct GPU_CHANNEL_FORMAT_DESC*, size_t, size_t, unsigned int)) dlsym(RTLD_NEXT, GPU_SYM(MallocArray));
	REAL_GPU(FreeArray)                = (GPU_ERROR_T(*)(GPU_ARRAY_T))                                                    dlsym(RTLD_NEXT, GPU_SYM(FreeArray));
	REAL_GPU(MallocHost)               = (GPU_ERROR_T(*)(void**, size_t))                                                 dlsym(RTLD_NEXT, GPU_SYM(MallocHost));
	REAL_GPU(FreeHost)                 = (GPU_ERROR_T(*)(void*))                                                          dlsym(RTLD_NEXT, GPU_SYM(FreeHost));
	REAL_GPU(HostAlloc)                = (GPU_ERROR_T(*)(void**, size_t, unsigned int))                                   dlsym(RTLD_NEXT, GPU_SYM(HostAlloc));
	REAL_GPU(Memset)                   = (GPU_ERROR_T(*)(void*, int, size_t))                                             dlsym(RTLD_NEXT, GPU_SYM(Memset));
	REAL_GPU(MallocManaged)            = (GPU_ERROR_T(*)(void**, size_t, unsigned int))                                   dlsym(RTLD_NEXT, GPU_SYM(MallocManaged));
	REAL_GPU(LaunchKernel)             = (GPU_ERROR_T(*)(const void*, dim3, dim3, void**, size_t, GPU_STREAM_T))          dlsym(RTLD_NEXT, GPU_SYM(LaunchKernel));
#else
	fprintf(stderr, PACKAGE_NAME": Warning! GPU instrumentation requires linking with shared library!\n");
#endif /* PIC */
}

/*
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
*/

#if defined(PIC)

GPU_ERROR_T GPU(Launch)(const void *func)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(Launch));
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s params %p\n", THREADID, __func__, func);
#endif
	if (REAL_GPU(Launch) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Launch_Entry((GPU_FUNCTION_T)(uintptr_t)func, 0, 0, 0, NULL, NULL);
		res = REAL_GPU(Launch)(func);
		Probe_Gpu_Launch_Exit(NULL);
	}
	else if (REAL_GPU(Launch) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(Launch)(func);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(ConfigureCall)(dim3 p1, dim3 p2, size_t p3, GPU_STREAM_T p4)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(ConfigureCall));
#endif
	if (REAL_GPU(ConfigureCall) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_ConfigureCall_Entry();
		res = REAL_GPU(ConfigureCall)(p1, p2, p3, p4);
		Probe_Gpu_ConfigureCall_Exit();
	}
	else if (REAL_GPU(ConfigureCall) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(ConfigureCall)(p1, p2, p3, p4);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(StreamCreate)(GPU_STREAM_T *pStream)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(StreamCreate));
#endif
	if (REAL_GPU(StreamCreate) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_StreamCreate_Entry();
		res = REAL_GPU(StreamCreate)(pStream);
		Probe_Gpu_StreamCreate_Exit();
	}
	else if (REAL_GPU(StreamCreate) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(StreamCreate)(pStream);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(StreamCreateWithFlags)(GPU_STREAM_T *pStream, unsigned int flags)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(StreamCreateWithFlags));
#endif
	if (REAL_GPU(StreamCreateWithFlags) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_StreamCreate_Entry();
		res = REAL_GPU(StreamCreateWithFlags)(pStream, flags);
		Probe_Gpu_StreamCreate_Exit();
	}
	else if (REAL_GPU(StreamCreateWithFlags) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(StreamCreateWithFlags)(pStream, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(StreamCreateWithPriority)(GPU_STREAM_T *pStream, unsigned int flags, int priority)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(StreamCreateWithPriority));
#endif
	if (REAL_GPU(StreamCreateWithPriority) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_StreamCreate_Entry();
		res = REAL_GPU(StreamCreateWithPriority)(pStream, flags, priority);
		Probe_Gpu_StreamCreate_Exit();
	}
	else if (REAL_GPU(StreamCreateWithPriority) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(StreamCreateWithPriority)(pStream, flags, priority);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(StreamDestroy)(GPU_STREAM_T stream)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(StreamDestroy));
#endif
	if (REAL_GPU(StreamDestroy) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_StreamDestroy_Entry(stream, NULL);
		res = REAL_GPU(StreamDestroy)(stream);
		Probe_Gpu_StreamDestroy_Exit();
	}
	else if (REAL_GPU(StreamDestroy) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(StreamDestroy)(stream);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(MemcpyAsync)(void *p1, const void *p2, size_t p3, GPU_MEMCPY_KIND_T p4, GPU_STREAM_T p5)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(MemcpyAsync));
#endif
	if (REAL_GPU(MemcpyAsync) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_MemcpyAsync_Entry(p1, p2, p3, p4, p5, NULL);
		res = REAL_GPU(MemcpyAsync)(p1, p2, p3, p4, p5);
		Probe_Gpu_MemcpyAsync_Exit(NULL);
	}
	else if (REAL_GPU(MemcpyAsync) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(MemcpyAsync)(p1, p2, p3, p4, p5);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(Memcpy)(void *p1, const void *p2, size_t p3, GPU_MEMCPY_KIND_T p4)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(Memcpy));
#endif
	if (REAL_GPU(Memcpy) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_gpuMemcpy_Enter(p1, p2, p3, p4, NULL);
		res = REAL_GPU(Memcpy)(p1, p2, p3, p4);
		Probe_gpuMemcpy_Exit(NULL);
	}
	else if (REAL_GPU(Memcpy) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(Memcpy)(p1, p2, p3, p4);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(ThreadSynchronize)(void)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(ThreadSynchronize));
#endif
	if (REAL_GPU(ThreadSynchronize) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_ThreadBarrier_Entry(NULL);
		res = REAL_GPU(ThreadSynchronize)();
		Probe_Gpu_ThreadBarrier_Exit();
	}
	else if (REAL_GPU(ThreadSynchronize) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(ThreadSynchronize)();
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(DeviceSynchronize)(void)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(DeviceSynchronize));
#endif
	if (REAL_GPU(DeviceSynchronize) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_ThreadBarrier_Entry(NULL);
		res = REAL_GPU(DeviceSynchronize)();
		Probe_Gpu_ThreadBarrier_Exit();
	}
	else if (REAL_GPU(DeviceSynchronize) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(DeviceSynchronize)();
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(StreamSynchronize)(GPU_STREAM_T p1)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(StreamSynchronize));
#endif
	if (REAL_GPU(StreamSynchronize) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_StreamBarrier_Entry(p1, NULL);
		res = REAL_GPU(StreamSynchronize)(p1);
		Probe_Gpu_StreamBarrier_Exit();
	}
	else if (REAL_GPU(StreamSynchronize) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(StreamSynchronize)(p1);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(ThreadExit)(void)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(ThreadExit));
#endif
	if (REAL_GPU(ThreadExit) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_ThreadExit_Entry();
		res = REAL_GPU(ThreadExit)();
		Probe_Gpu_ThreadExit_Exit(NULL);
	}
	else if (REAL_GPU(ThreadExit) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(ThreadExit)();
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(DeviceReset)(void)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(DeviceReset));
#endif
	if (REAL_GPU(DeviceReset) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_DeviceReset_Entry();
		res = REAL_GPU(DeviceReset)();
		Probe_Gpu_DeviceReset_Exit(NULL);
	}
	else if (REAL_GPU(DeviceReset) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(DeviceReset)();
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(Malloc)(void **devPtr, size_t size)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(Malloc));
#endif
	if (REAL_GPU(Malloc) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Malloc_Entry(GPUEV(MALLOC_VAL), (UINT64)devPtr, size);
		res = REAL_GPU(Malloc)(devPtr, size);
		Probe_Gpu_Malloc_Exit(GPUEV(MALLOC_VAL));
	}
	else if (REAL_GPU(Malloc) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(Malloc)(devPtr, size);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(MallocPitch)(void **devPtr, size_t *pitch, size_t width, size_t height)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(MallocPitch));
#endif
	if (REAL_GPU(MallocPitch) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Malloc_Entry(GPUEV(MALLOCPITCH_VAL), (UINT64)devPtr, width * height);
		res = REAL_GPU(MallocPitch)(devPtr, pitch, width, height);
		Probe_Gpu_Malloc_Exit(GPUEV(MALLOCPITCH_VAL));
	}
	else if (REAL_GPU(MallocPitch) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(MallocPitch)(devPtr, pitch, width, height);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(Free)(void *devPtr)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(Free));
#endif
	if (REAL_GPU(Free) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Free_Entry(GPUEV(FREE_VAL), (UINT64)devPtr);
		res = REAL_GPU(Free)(devPtr);
		Probe_Gpu_Free_Exit(GPUEV(FREE_VAL));
	}
	else if (REAL_GPU(Free) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(Free)(devPtr);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(MallocArray)(GPU_ARRAY_T *array, const struct GPU_CHANNEL_FORMAT_DESC *desc,
	size_t width, size_t height, unsigned int flags)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(MallocArray));
#endif
	if (REAL_GPU(MallocArray) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Malloc_Entry(GPUEV(MALLOCARRAY_VAL), (UINT64)array, width * height);
		res = REAL_GPU(MallocArray)(array, desc, width, height, flags);
		Probe_Gpu_Malloc_Exit(GPUEV(MALLOCARRAY_VAL));
	}
	else if (REAL_GPU(MallocArray) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(MallocArray)(array, desc, width, height, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(FreeArray)(GPU_ARRAY_T array)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(FreeArray));
#endif
	if (REAL_GPU(FreeArray) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Free_Entry(GPUEV(FREEARRAY_VAL), (UINT64)array);
		res = REAL_GPU(FreeArray)(array);
		Probe_Gpu_Free_Exit(GPUEV(FREEARRAY_VAL));
	}
	else if (REAL_GPU(FreeArray) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(FreeArray)(array);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(MallocHost)(void **ptr, size_t size)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(MallocHost));
#endif
	if (REAL_GPU(MallocHost) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Malloc_Entry(GPUEV(MALLOCHOST_VAL), (UINT64)ptr, size);
		res = REAL_GPU(MallocHost)(ptr, size);
		Probe_Gpu_Malloc_Exit(GPUEV(MALLOCHOST_VAL));
	}
	else if (REAL_GPU(MallocHost) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(MallocHost)(ptr, size);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(FreeHost)(void *ptr)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(FreeHost));
#endif
	if (REAL_GPU(FreeHost) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Free_Entry(GPUEV(FREEHOST_VAL), (UINT64)ptr);
		res = REAL_GPU(FreeHost)(ptr);
		Probe_Gpu_Free_Exit(GPUEV(FREEHOST_VAL));
	}
	else if (REAL_GPU(FreeHost) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(FreeHost)(ptr);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(HostAlloc)(void **pHost, size_t size, unsigned int flags)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(HostAlloc));
#endif
	if (REAL_GPU(HostAlloc) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_HostAlloc_Entry((UINT64)pHost, size);
		res = REAL_GPU(HostAlloc)(pHost, size, flags);
		Probe_Gpu_HostAlloc_Exit();
	}
	else if (REAL_GPU(HostAlloc) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(HostAlloc)(pHost, size, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

GPU_ERROR_T GPU(Memset)(void *devPtr, int value, size_t count)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(Memset));
#endif
	if (REAL_GPU(Memset) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Memset_Entry((UINT64)devPtr, count);
		res = REAL_GPU(Memset)(devPtr, value, count);
		Probe_Gpu_Memset_Exit();
	}
	else if (REAL_GPU(Memset) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(Memset)(devPtr, value, count);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

/* v6.0 */
GPU_ERROR_T GPU(MallocManaged)(void **devPtr, size_t size, unsigned int flags)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(MallocManaged));
#endif
	if (REAL_GPU(MallocManaged) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Malloc_Entry(GPUEV(MALLOC_VAL), (UINT64)devPtr, size);
		res = REAL_GPU(MallocManaged)(devPtr, size, flags);
		Probe_Gpu_Malloc_Exit(GPUEV(MALLOC_VAL));
	}
	else if (REAL_GPU(MallocManaged) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(MallocManaged)(devPtr, size, flags);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs. Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

/* v7.0 */
GPU_ERROR_T GPU(LaunchKernel)(const void *func, dim3 gridDim, dim3 blockDim,
	void **args, size_t sharedMem, GPU_STREAM_T stream)
{
	GPU_ERROR_T res;
	Backend_Enter_Instrumentation();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME": THREAD %d %s is at %p\n", THREADID, __func__, REAL_GPU(LaunchKernel));
#endif
	if (REAL_GPU(LaunchKernel) != NULL && mpitrace_on && Extrae_get_trace_GPU())
	{
		Probe_Gpu_Launch_Entry(
			(GPU_FUNCTION_T)(uintptr_t)func,
			gridDim.x * gridDim.y * gridDim.z,
			blockDim.x * blockDim.y * blockDim.z,
			sharedMem, stream, NULL);
		res = REAL_GPU(LaunchKernel)(func, gridDim, blockDim, args, sharedMem, stream);
		Probe_Gpu_Launch_Exit(NULL);
	}
	else if (REAL_GPU(LaunchKernel) != NULL && !(mpitrace_on && Extrae_get_trace_GPU()))
	{
		res = REAL_GPU(LaunchKernel)(func, gridDim, blockDim, args, sharedMem, stream);
	}
	else
	{
		fprintf(stderr, "Unable to find %s in DSOs! Dying...\n", __func__);
		exit(0);
	}
	Backend_Leave_Instrumentation();
	return res;
}

#endif /* PIC */