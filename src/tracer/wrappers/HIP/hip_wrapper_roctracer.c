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

#include <roctracer/roctracer.h>
#include <roctracer/roctracer_hip.h>
#include "gpu_common.h"
#include "gpu_probe.h"

#define LOG_OTHER_DOMAIN_UNTRACKED_CALLBACKS 0

#include "hip_wrapper_roctracer.h"

#define MAX_HIP_API_ID 1024
static bool cid_filter[MAX_HIP_API_ID] = {false};

#define SET_CID_FILTER(cid)   (cid_filter[cid] = true)
#define IS_CID_FILTERED(cid)  (cid_filter[cid])

static void 
Extrae_HIP_API_disable_calls(void) {
    SET_CID_FILTER(HIP_API_ID___hipPushCallConfiguration);
    SET_CID_FILTER(HIP_API_ID___hipPopCallConfiguration);
    SET_CID_FILTER(HIP_API_ID_hipRuntimeGetVersion);
    SET_CID_FILTER(HIP_API_ID_hipDeviceGet);
    SET_CID_FILTER(HIP_API_ID_hipGetLastError);
    SET_CID_FILTER(HIP_API_ID_hipDeviceGetName);
    SET_CID_FILTER(HIP_API_ID_hipGetDeviceCount);
    SET_CID_FILTER(HIP_API_ID_hipEventCreate);
    SET_CID_FILTER(HIP_API_ID_hipEventDestroy);
    SET_CID_FILTER(HIP_API_ID_hipEventElapsedTime);
    SET_CID_FILTER(HIP_API_ID_hipHostRegister);
    SET_CID_FILTER(HIP_API_ID_hipFuncSetCacheConfig);
    SET_CID_FILTER(HIP_API_ID_hipGetDevicePropertiesR0600);
    SET_CID_FILTER(HIP_API_ID_hipDeviceGetAttribute);
    SET_CID_FILTER(HIP_API_ID_hipFuncGetAttribute);
    SET_CID_FILTER(HIP_API_ID_hipFuncGetAttributes);
    SET_CID_FILTER(HIP_API_ID_hipDeviceComputeCapability);
    SET_CID_FILTER(HIP_API_ID_hipPointerGetAttributes);
    SET_CID_FILTER(HIP_API_ID_hipModuleGetGlobal);
    SET_CID_FILTER(HIP_API_ID_hipModuleLoadData);
    SET_CID_FILTER(HIP_API_ID_hipModuleLoad);
    SET_CID_FILTER(HIP_API_ID_hipModuleUnload);
    SET_CID_FILTER(HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize);
    SET_CID_FILTER(HIP_API_ID_hipGetSymbolAddress);
    SET_CID_FILTER(HIP_API_ID_hipModuleLaunchCooperativeKernel);
    SET_CID_FILTER(HIP_API_ID_hipExtLaunchKernel);
    SET_CID_FILTER(HIP_API_ID_hipSetDevice);
}

static void 
Extrae_ROCTRACER_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) 
{

    UNREFERENCED_PARAMETER(arg);
    UNREFERENCED_PARAMETER(callback_data);

    if (!mpitrace_on || !Extrae_get_trace_GPU() || callback_data == NULL || IS_CID_FILTERED(cid))
        return;

    int ret = 0;
    const hip_api_data_t* p = (const hip_api_data_t*)callback_data;

    if (p->phase == ACTIVITY_API_PHASE_ENTER) 
        Backend_Enter_Instrumentation();

    if (EXTRAE_ON() && Extrae_get_trace_GPU() && Backend_Get_InstrumentationLevel() == 1) {
        switch (cid) {
            // DriverAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020 equivalent
            // CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc equivalent
            case HIP_API_ID_hipHostAlloc:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_HostAlloc_Entry(p->args.hipHostAlloc.ptr, 
                        p->args.hipHostAlloc.size);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT) 
                {
                    Probe_Gpu_HostAlloc_Exit();
                }
                ret = 1;
            }
            break;

            // DriverAPI on cuda
            // CUPTI_DRIVER_TRACE_CBID_cuStreamCreate equivalent
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020 equivalent
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000 equivalent
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050 equivalent
            case HIP_API_ID_hipStreamCreateWithFlags:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_StreamCreate_Entry();
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_StreamCreate_Exit();
                }
                ret = 1;
            }
            break;

            // DriverAPI on cuda
            // CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize equivalent
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020 equivalent
            case HIP_API_ID_hipStreamSynchronize:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_StreamBarrier_Entry(p->args.hipStreamSynchronize.stream, NULL);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_StreamBarrier_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020 equivalent
            case HIP_API_ID_hipConfigureCall:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_ConfigureCall_Entry();
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT) 
                {
                    Probe_Gpu_ConfigureCall_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020 equivalent
            case HIP_API_ID_hipMalloc:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_Malloc_Entry(HIPMALLOC_VAL, p->args.hipMalloc.ptr, p->args.hipMalloc.size);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT) 
                {
                    Probe_Gpu_Malloc_Exit(HIPMALLOC_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020 equivalent
            case HIP_API_ID_hipMallocPitch:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Malloc_Entry(HIPMALLOCPITCH_VAL, 
                        p->args.hipMallocPitch.ptr, 
                        p->args.hipMallocPitch.width * p->args.hipMallocPitch.height);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Malloc_Exit(HIPMALLOCPITCH_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020 equivalent
            case HIP_API_ID_hipFree:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Free_Entry(HIPFREE_VAL, 
                        p->args.hipFree.ptr);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Free_Exit(HIPFREE_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020 equivalent
            case HIP_API_ID_hipMallocArray: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_Malloc_Entry(HIPMALLOCPITCH_VAL, 
                        p->args.hipMallocPitch.ptr, 
                        p->args.hipMallocPitch.width * p->args.hipMallocPitch.height);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Malloc_Exit(HIPMALLOCPITCH_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020 equivalent
            case HIP_API_ID_hipFreeArray: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Free_Entry(HIPFREEARRAY_VAL, 
                        (void*)p->args.hipFreeArray.array);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Free_Exit(HIPFREEARRAY_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020 equivalent
            case HIP_API_ID_hipHostMalloc: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_Malloc_Entry(HIPMALLOCHOST_VAL, 
                        p->args.hipHostMalloc.ptr, 
                        p->args.hipHostMalloc.size);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT) 
                {
                    Probe_Gpu_Malloc_Exit(HIPMALLOCHOST_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020 equivalent
            case HIP_API_ID_hipHostFree: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_Free_Entry(HIPFREEHOST_VAL, 
                        p->args.hipHostFree.ptr);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT) 
                {
                    Probe_Gpu_Free_Exit(HIPFREEHOST_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 equivalent
            case HIP_API_ID_hipMemcpy:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_gpuMemcpy_Enter(p->args.hipMemcpy.dst,
                        p->args.hipMemcpy.src,
                        p->args.hipMemcpy.sizeBytes,
                        p->args.hipMemcpy.kind,
                        NULL);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_gpuMemcpy_Exit(NULL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyToSymbol_v3020 equivalent
            case HIP_API_ID_hipMemcpyToSymbol: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_gpuMemcpyToSymbol_Enter(NULL, 
                        p->args.hipMemcpyToSymbol.src, 
                        p->args.hipMemcpyToSymbol.sizeBytes, 
                        p->args.hipMemcpyToSymbol.kind, 
                        NULL);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_gpuMemcpyToSymbol_Exit(NULL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyFromSymbol_v3020 equivalent
            case HIP_API_ID_hipMemcpyFromSymbol: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_gpuMemcpyFromSymbol_Enter(p->args.hipMemcpyFromSymbol.dst, 
                        NULL, 
                        p->args.hipMemcpyFromSymbol.sizeBytes, 
                        p->args.hipMemcpyFromSymbol.kind, 
                        NULL);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_gpuMemcpyFromSymbol_Exit(NULL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020 equivalent
            case HIP_API_ID_hipMemcpyAsync:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_MemcpyAsync_Entry(p->args.hipMemcpyAsync.dst, 
                        p->args.hipMemcpyAsync.src,
                        p->args.hipMemcpyAsync.sizeBytes, 
                        p->args.hipMemcpyAsync.kind,
                        p->args.hipMemcpyAsync.stream, 
                        NULL);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_MemcpyAsync_Exit(NULL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020 equivalent
            case HIP_API_ID_hipMemset:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Memset_Entry(p->args.hipMemset.dst, 
                        p->args.hipMemset.sizeBytes);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Memset_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020 equivalent
            case HIP_API_ID_hipMemsetAsync:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_MemsetAsync_Entry(p->args.hipMemsetAsync.dst, p->args.hipMemsetAsync.sizeBytes);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_MemsetAsync_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020 equivalent
            case HIP_API_ID_hipDeviceSynchronize:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_ThreadBarrier_Entry(NULL);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_ThreadBarrier_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_DRIVER_TRACE_CBID_cuStreamCreate equivalent
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020 equivalent
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000 equivalent
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050 equivalent
            case HIP_API_ID_hipStreamCreate:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_StreamCreate_Entry();
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_StreamCreate_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020 / v5050 equivalent
            case HIP_API_ID_hipStreamDestroy:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_StreamDestroy_Entry((hipStream_t)p->args.hipStreamDestroy.stream, 
                        NULL);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_StreamDestroy_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020 equivalent
            case HIP_API_ID_hipEventRecord:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_EventRecord_Entry((hipEvent_t)p->args.hipEventRecord.event, 
                        (hipStream_t)p->args.hipEventRecord.stream, 
                        NULL);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT) 
                {
                    Probe_Gpu_EventRecord_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020 equivalent
            case HIP_API_ID_hipEventSynchronize:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_EventSynchronize_Entry((hipEvent_t)p->args.hipEventSynchronize.event);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_EventSynchronize_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020 equivalent
            case HIP_API_ID_hipMalloc3D: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Malloc_Entry(HIPMALLOC3D_VAL, NULL, p->args.hipMalloc3D.extent.width * p->args.hipMalloc3D.extent.height * p->args.hipMalloc3D.extent.depth);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Malloc_Exit(HIPMALLOC3D_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            //CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020 equivalent
            case HIP_API_ID_hipMalloc3DArray: 
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_Malloc_Entry(HIPMALLOC3DARRAY_VAL, 
                        (void*)p->args.hipMalloc3DArray.array, 
                        p->args.hipMalloc3DArray.extent.width * p->args.hipMalloc3DArray.extent.height * p->args.hipMalloc3DArray.extent.depth);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Malloc_Exit(HIPMALLOC3DARRAY_VAL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020 equivalent
            case HIP_API_ID_hipDeviceReset:
            {
                if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_DeviceReset_Exit(NULL);
                }
                else if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_DeviceReset_Entry();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020 equivalent
            case HIP_API_ID_hipStreamWaitEvent:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_StreamWaitEvent_Entry(p->args.hipStreamWaitEvent.event);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_StreamWaitEvent_Exit();
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050 equivalent
            case HIP_API_ID_hipStreamCreateWithPriority:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_StreamCreate_Entry();
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_StreamCreate_Exit();
                }
                ret = 1;
            }
            break;
            
            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000 equivalent
            case HIP_API_ID_hipMallocManaged:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Malloc_Entry(HIPMALLOC_VAL, p->args.hipMallocManaged.dev_ptr, p->args.hipMallocManaged.size);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Malloc_Exit(HIPMALLOC_VAL);
                }
                ret = 1;
            }
            break;

            // DriverAPI on cuda
            // CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel equivalent
            case HIP_API_ID_hipModuleLaunchKernel:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Launch_Entry(p->args.hipModuleLaunchKernel.f,
                        p->args.hipModuleLaunchKernel.gridDimX * p->args.hipModuleLaunchKernel.gridDimY * p->args.hipModuleLaunchKernel.gridDimZ,
                        p->args.hipModuleLaunchKernel.blockDimX * p->args.hipModuleLaunchKernel.blockDimY * p->args.hipModuleLaunchKernel.blockDimZ,
                        p->args.hipModuleLaunchKernel.sharedMemBytes,
                        p->args.hipModuleLaunchKernel.stream,
                        NULL);
                }
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Launch_Exit(NULL);
                }
                ret = 1;
            }
            break;

            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 equivalent
            case HIP_API_ID_hipLaunchKernel:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER) 
                {
                    Probe_Gpu_Launch_Entry(p->args.hipLaunchKernel.function_address, 
                        p->args.hipLaunchKernel.numBlocks.x * p->args.hipLaunchKernel.numBlocks.y * p->args.hipLaunchKernel.numBlocks.z,
                        p->args.hipLaunchKernel.dimBlocks.x * p->args.hipLaunchKernel.dimBlocks.y * p->args.hipLaunchKernel.dimBlocks.z,
                        p->args.hipLaunchKernel.sharedMemBytes, 
                        p->args.hipLaunchKernel.stream, 
                        NULL);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Launch_Exit(NULL);
                }
                ret = 1;
            }
            break;
            
            // RUNTIMEAPI on cuda
            // CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000 equivalent
            case HIP_API_ID_hipLaunchCooperativeKernel:
            {
                if (p->phase == ACTIVITY_API_PHASE_ENTER)
                {
                    Probe_Gpu_Launch_Entry(p->args.hipLaunchCooperativeKernel.f,
                        p->args.hipLaunchCooperativeKernel.gridDim.x * p->args.hipLaunchCooperativeKernel.gridDim.y * p->args.hipLaunchCooperativeKernel.gridDim.z,
                        p->args.hipLaunchCooperativeKernel.blockDimX.x * p->args.hipLaunchCooperativeKernel.blockDimX.y * p->args.hipLaunchCooperativeKernel.blockDimX.z,
                        p->args.hipLaunchCooperativeKernel.sharedMemBytes,
                        p->args.hipLaunchCooperativeKernel.stream,
                        NULL);
                } 
                else if (p->phase == ACTIVITY_API_PHASE_EXIT)
                {
                    Probe_Gpu_Launch_Exit(NULL);
                }
                ret = 1;
            }
            break;
            default:
            {
                #if LOG_OTHER_DOMAIN_UNTRACKED_CALLBACKS
                    const char* name = roctracer_op_string(domain, cid, 0);
                    fprintf(stderr, "Untracked Domain HIP event (domain=%u cid=%u name=%s)\n", domain, cid, name);
                    fflush(stderr);
                #endif
                ret = -1;
            }
            break;
        }
    }

    if (p->phase == ACTIVITY_API_PHASE_EXIT)
        Backend_Leave_Instrumentation();

}

void Extrae_HIP_init(int rank) {

    UNREFERENCED_PARAMETER(rank);

    Extrae_HIP_API_disable_calls();

    /* Equivalent to cuptiEnableAllDomains(...) + disable calls: Here we enable HIP API domain callback and apply cid filter above. */
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, Extrae_ROCTRACER_callback, NULL);

}