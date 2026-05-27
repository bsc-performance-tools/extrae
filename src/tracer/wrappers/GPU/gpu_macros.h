#ifndef GPU_MACROS_H_INCLUDED

#include "common.h"
#define GPU_MACROS_H_INCLUDED

/* ══════════════════════════════════════════════════════════════════════════
 * 0. Backend includes
 * ══════════════════════════════════════════════════════════════════════════ */

#if defined(CUDA_SUPPORT)
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(HIP_SUPPORT)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

#if !defined(HIP_SUPPORT) && !defined(CUDA_SUPPORT)
#  error "gpu_common.h requires -DUSE_HIP or -DUSE_CUDA"
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * 1. Concatenation macros — core of GPU(a)
 *
 *  GPU(name)  →  HIPname   / CUDAname       (no underscore)
 *
 *  Two levels of expansion are REQUIRED so the preprocessor
 *  expands GPU_BACKEND before concatenating.
 * ══════════════════════════════════════════════════════════════════════════ */
#if defined(CUDA_SUPPORT)
#  define GPU_BACKEND        cuda
#  define GPU_BACKEND_UPPER  CUDA
#elif defined(HIP_SUPPORT)
#  define GPU_BACKEND        hip
#  define GPU_BACKEND_UPPER  HIP
#endif

#define _GPU_CONCAT(a, b)    a##b
#define _GPU_EXPAND(a, b)    _GPU_CONCAT(a, b)

/* ── Function pointer prefix — real_hipLaunch / real_cudaLaunch ── */
#define REAL_GPU(name) _GPU_EXPAND(real_GPU_BACKEND,  name)

/* ── Runtime API (lowercase) — hipStream_t, cudaGetDevice(), ... ── */
#define GPU(name)      _GPU_EXPAND(GPU_BACKEND,       name)

/* ── Extrae constants (uppercase) — HIPCALL_EV, CUDAMEMCPY_VAL, ... ── */
#define GPUEV(name)    _GPU_EXPAND(GPU_BACKEND_UPPER, name)

/* ── String version of GPU API name for dlsym ── */
#define _GPU_STR(x)   #x
#define _GPU_XSTR(x)  _GPU_STR(x)
#define GPU_SYM(name) _GPU_XSTR(GPU(name)) 

/* ══════════════════════════════════════════════════════════════════════════
 * 2. Unified types  (replace the two duplicated #ifdef blocks)
 * ══════════════════════════════════════════════════════════════════════════ */
#define GPU_STREAM_T                 GPU(Stream_t)                    /* hipStream_t   / cudaStream_t   */
#define GPU_EVENT_T                  GPU(Event_t)                     /* hipEvent_t    / cudaEvent_t    */
#define GPU_MEMCPY_KIND_T            enum GPU(MemcpyKind)             /* hipMemcpyKind / cudaMemcpyKind */
#define GPU_FUNCTION_T               GPU(Function_t)                  /* hipFunction_t / cudaFunction_t */
#define GPU_STREAM_DEFAULT           GPU(StreamDefault)               /* hipStreamDefault / cudaStreamDefault */
#define GPU_MEMCPY_H2D_T             GPU(MemcpyHostToDevice)
#define GPU_MEMCPY_D2H_T             GPU(MemcpyDeviceToHost)
#define GPU_EVENT                    GPU(Event)                       /* hipEvent      / cudaEvent (probe prefix) */
#define GPU_ARRAY_T                  GPU(Array_t)                     /* hipArray_t    / cudaArray_t */
#define GPU_CHANNEL_FORMAT_DESC      GPU(ChannelFormatDesc)           /* hipChannelFormatDesc / cudaChannelFormatDesc */
#define GPU_ERROR_T                  GPU(Error_t)                     /* cudaError_t   / hipError_t */

/*
 *     EXCEPTION — GPU_CONTEXT_T cannot be unified cleanly:
 *     HIP  → hipCtx_t
 *     CUDA → CUcontext
 */
#if defined(CUDA_SUPPORT)
#  define GPU_CONTEXT_T  CUcontext   /* CUDA driver API — does not follow the cuda* pattern */
#elif defined(HIP_SUPPORT)
#  define GPU_CONTEXT_T  hipCtx_t
#endif

/* ══════════════════════════════════════════════════════════════════════════
 * 3. Unified API function wrappers
 * ══════════════════════════════════════════════════════════════════════════ */
#define GPU_EVENT_CREATE_WITH_FLAGS(ev, flags) GPU(EventCreateWithFlags)((ev),   (flags))
#define GPU_EVENT_DESTROY(ev)                  GPU(EventDestroy)((ev))
#define GPU_EVENT_RECORD(ev, stream)           GPU(EventRecord)((ev),            (stream))
#define GPU_EVENT_SYNCHRONIZE(ev)              GPU(EventSynchronize)((ev))
#define GPU_EVENT_ELAPSED_TIME(ms, a, b)       GPU(EventElapsedTime)((ms),(a),   (b))
#define GPU_GET_DEVICE_COUNT(count)            GPU(GetDeviceCount)((count))
#define GPU_GET_DEVICE(devptr)                 GPU(GetDevice)((devptr))

/* ══════════════════════════════════════════════════════════════════════════
 * 4. Error checking — ASYMMETRIC between HIP and CUDA (driver vs runtime)
 *  
 *    HIP:  hipError_t + hipSuccess  (runtime and driver use the same type)
 *    CUDA: cudaError_t + cudaSuccess  (runtime API)
 *          CUresult   + CUDA_SUCCESS  (driver API)
 *  
 * ══════════════════════════════════════════════════════════════════════════ */
#if defined(CUDA_SUPPORT)

#define GPU_DRIVER_CHECK(call)                                                 \
    do {                                                                       \
        CUresult _r = (call);                                                  \
        if (_r != CUDA_SUCCESS) {                                              \
            const char *_s = NULL;                                             \
            cuGetErrorString(_r, &_s);                                         \
            fprintf(stderr,                                                    \
                "[CUDA DRIVER ERROR] '%s'\n  at %s:%d\n  %s\n",                \
                #call, __FILE__, __LINE__,                                     \
                _s ? _s : "Unknown");                                          \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define GPU_RUNTIME_CHECK(call)                                                \
    do {                                                                       \
        cudaError_t _r = (call);                                               \
        if (_r != cudaSuccess) {                                               \
            fprintf(stderr,                                                    \
                "[CUDA RUNTIME ERROR] '%s'\n  at %s:%d\n  %s: %s\n",           \
                #call, __FILE__, __LINE__,                                     \
                cudaGetErrorName(_r), cudaGetErrorString(_r));                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/*
 * CUDA driver API requires explicit context push/pop to ensure the correct
 * device is active when retrieving the device ID from a driver-level context.
 */
#define GPU_GET_DEVICE_SAFE(ctx, device_id)                                    \
    do {                                                                       \
        if ((ctx) != NULL) {                                                   \
            GPU_DRIVER_CHECK(cuCtxPushCurrent(ctx));                           \
            GPU_RUNTIME_CHECK(cudaGetDevice(&(device_id)));                    \
            GPU_DRIVER_CHECK(cuCtxPopCurrent(&(ctx)));                         \
        } else {                                                               \
            GPU_RUNTIME_CHECK(cudaGetDevice(&(device_id)));                    \
        }                                                                      \
    } while (0)

#endif /* CUDA_SUPPORT */

#if defined(HIP_SUPPORT)

#define GPU_DRIVER_CHECK(call)                                                 \
    do {                                                                       \
        hipError_t _r = (call);                                                \
        if (_r != hipSuccess) {                                                \
            fprintf(stderr,                                                    \
                "[GPU DRIVER ERROR] '%s'\n  at %s:%d\n  %s: %s\n",             \
                #call, __FILE__, __LINE__,                                     \
                hipGetErrorName(_r), hipGetErrorString(_r));                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define GPU_RUNTIME_CHECK(call)  GPU_DRIVER_CHECK(call)  /* HIP: same type for runtime and driver */

/*
 * ROCm 6.x: hipCtxPushCurrent/hipCtxPopCurrent are deprecated.
 * Device is implicit in the calling thread — hipGetDevice() suffices.
 * The ctx parameter is kept in the signature for CUDA ABI compatibility
 * but is ignored on the HIP path.
 */
#define GPU_GET_DEVICE_SAFE(ctx, device_id)                                    \
    do {                                                                       \
        (void)(ctx);                                                           \
        GPU_RUNTIME_CHECK(hipGetDevice(&(device_id)));                         \
    } while (0)

#endif /* HIP_SUPPORT */

/* ══════════════════════════════════════════════════════════════════════════
 * 5. GPU trace macros
 *    Originally defined in gpu_macros.h — kept here as the single
 *    definition point to avoid dependency on the legacy header.
 * ══════════════════════════════════════════════════════════════════════════ */
#define TRACE_GPU_EVENT(_thread, _evttime, _evttype, _evtvalue, _evtbegin, _evtsize)                      \
    {                                                                                                     \
        int _thread_id = _thread;                                                                         \
        event_t _evt;                                                                                     \
        if (tracejant && TracingBitmap[TASKID])                                                           \
        {                                                                                                 \
            _evt.time                    = _evttime;                                                      \
            _evt.event                   = _evttype;                                                      \
            _evt.value                   = _evtvalue;                                                     \
            _evt.param.gpu_param.begin   = _evtbegin;                                                     \
            _evt.param.gpu_param.memSize = _evtsize;                                                      \
            HARDWARE_COUNTERS_READ(_thread_id, _evt, FALSE);                                              \
            BUFFER_INSERT(_thread_id, TRACING_BUFFER(_thread_id), _evt);                                  \
        }                                                                                                 \
    }

#define TRACE_GPU_KERNEL_EVENT(_thread, _evttime, _evttype, _evtvalue, _blockspergrid, _threadsperblock)  \
    {                                                                                                     \
        event_t _evt;                                                                                     \
        if (tracejant && TracingBitmap[TASKID])                                                           \
        {                                                                                                 \
            _evt.time                    = _evttime;                                                      \
            _evt.event                   = _evttype;                                                      \
            _evt.value                   = _evtvalue;                                                     \
            _evt.param.gpu_param.gridSize  = _blockspergrid;                                              \
            _evt.param.gpu_param.blockSize = _threadsperblock;                                            \
            HARDWARE_COUNTERS_READ(_thread, _evt, FALSE);                                                 \
            BUFFER_INSERT(_thread, TRACING_BUFFER(_thread), _evt);                                        \
        }                                                                                                 \
    }

#endif /* GPU_MACROS_H_INCLUDED */