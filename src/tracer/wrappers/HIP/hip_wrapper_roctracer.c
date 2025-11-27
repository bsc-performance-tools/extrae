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
#include "hip_common.h"
#include "hip_probe.h"

#define LOG_UNTRACKED_CALLBACKS 1
#define LOG_OTHER_DOMAIN_UNTRACKED_CALLBACKS 1

#include "hip_wrapper_roctracer.h"

#define MAX_HIP_API_ID 1024
static uint8_t cid_filter[MAX_HIP_API_ID / 8] = {0};

#define SET_CID_FILTER(cid)   (cid_filter[(cid) / 8] |= (1 << ((cid) % 8)))
#define CLEAR_CID_FILTER(cid) (cid_filter[(cid) / 8] &= ~(1 << ((cid) % 8)))
#define IS_CID_FILTERED(cid)  (cid_filter[(cid) / 8] &  (1 << ((cid) % 8)))

static void Extrae_HIP_API_disable_calls(void)
{
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
    SET_CID_FILTER(HIP_API_ID_hipStreamWaitEvent);
    SET_CID_FILTER(HIP_API_ID_hipPointerGetAttributes);
    SET_CID_FILTER(HIP_API_ID_hipModuleGetGlobal);
    SET_CID_FILTER(HIP_API_ID_hipModuleLoadData);
    SET_CID_FILTER(HIP_API_ID_hipModuleLoad);
    SET_CID_FILTER(HIP_API_ID_hipModuleUnload);
    SET_CID_FILTER(HIP_API_ID_hipModuleOccupancyMaxPotentialBlockSize);
    SET_CID_FILTER(HIP_API_ID_hipGetSymbolAddress);
    SET_CID_FILTER(HIP_API_ID_hipDeviceReset);
    SET_CID_FILTER(HIP_API_ID_hipModuleLaunchCooperativeKernel);
    SET_CID_FILTER(HIP_API_ID_hipExtLaunchKernel);
}

static int 
Extrae_HIP_API_callback(const hip_api_data_t* p, uint32_t cid, const void* callback_data, void* arg) 
{
    int ret = 0;

    switch (cid)
    {
        case HIP_API_ID_hipMalloc:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMalloc_Enter(HIPMALLOC_VAL, p->args.hipMalloc.ptr, p->args.hipMalloc.size);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMalloc_Exit(HIPMALLOC_VAL);
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipHostMalloc:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMalloc_Enter(HIPMALLOC_VAL, p->args.hipHostMalloc.ptr, p->args.hipHostMalloc.size);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMalloc_Exit(HIPMALLOC_VAL);
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipHostAlloc:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipHostAlloc_Enter(p->args.hipHostAlloc.ptr, p->args.hipHostAlloc.size);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipHostAlloc_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipMemcpy:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMemcpy_Enter(p->args.hipMemcpy.dst, 
                    p->args.hipMemcpy.src, 
                    p->args.hipMemcpy.sizeBytes, 
                    p->args.hipMemcpy.kind);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMemcpy_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipMemcpyAsync:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMemcpyAsync_Enter(p->args.hipMemcpyAsync.dst, 
                    p->args.hipMemcpyAsync.src,
                    p->args.hipMemcpyAsync.sizeBytes,
                    p->args.hipMemcpyAsync.kind,
                    p->args.hipMemcpyAsync.stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMemcpyAsync_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipMemcpyHtoD:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMemcpyHtoD_Enter(p->args.hipMemcpyHtoD.dst, 
                    p->args.hipMemcpyHtoD.src, 
                    p->args.hipMemcpyHtoD.sizeBytes);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMemcpyHtoD_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipMemcpyHtoDAsync:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMemcpyHtoDAsync_Enter(p->args.hipMemcpyHtoDAsync.dst, 
                    p->args.hipMemcpyHtoDAsync.src, 
                    p->args.hipMemcpyHtoDAsync.sizeBytes,
                    p->args.hipMemcpyHtoDAsync.stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMemcpyHtoDAsync_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipMemcpyDtoH:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMemcpyDtoH_Enter(p->args.hipMemcpyHtoD.dst, 
                    p->args.hipMemcpyHtoD.src, 
                    p->args.hipMemcpyHtoD.sizeBytes);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMemcpyDtoH_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipMemcpyDtoHAsync:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMemcpyDtoHAsync_Enter(p->args.hipMemcpyHtoDAsync.dst, 
                    p->args.hipMemcpyHtoDAsync.src, 
                    p->args.hipMemcpyHtoDAsync.sizeBytes,
                    p->args.hipMemcpyHtoDAsync.stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMemcpyDtoHAsync_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipFree:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipFree_Enter(HIPFREE_VAL, p->args.hipFree.ptr);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipFree_Exit(HIPFREE_VAL);
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipMemset:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipMemset_Enter(p->args.hipMemset.dst, p->args.hipMemset.sizeBytes);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipMemset_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipEventRecord:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipEventRecord_Enter();
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipEventRecord_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipEventSynchronize:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipEventSynchronize_Enter();
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipEventSynchronize_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipStreamCreate:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipStreamCreate_Enter(p->args.hipStreamCreate.stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipStreamCreate_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipStreamDestroy:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipStreamDestroy_Enter(p->args.hipStreamDestroy.stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipStreamDestroy_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipStreamSynchronize:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipStreamSynchronize_Enter(p->args.hipStreamSynchronize.stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipStreamSynchronize_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipDeviceSynchronize:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                Extrae_hipDeviceSynchronize_Enter();
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipDeviceSynchronize_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipModuleGetFunction:
        {
            if (p->phase == ACTIVITY_API_PHASE_EXIT) 
            {
                hipFunction_t* outp = (hipFunction_t*) p->args.hipModuleGetFunction.function;
                hipFunction_t f     = outp ? *outp : NULL;
                const char*   k     = (const char*) p->args.hipModuleGetFunction.kname;
                if(f)
                {
                    hipmap_put_safe(f, k);
                }
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipModuleLaunchKernel:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                hipFunction_t f = (hipFunction_t)p->args.hipModuleLaunchKernel.f;
                unsigned gridDim = (int)p->args.hipModuleLaunchKernel.gridDimX * (int)p->args.hipModuleLaunchKernel.gridDimY * (int)p->args.hipModuleLaunchKernel.gridDimZ;
                unsigned blockDim = (int)p->args.hipModuleLaunchKernel.blockDimX * (int)p->args.hipModuleLaunchKernel.blockDimY * (int)p->args.hipModuleLaunchKernel.blockDimZ;
                size_t sharedMemBytes = (size_t)p->args.hipModuleLaunchKernel.sharedMemBytes;
                hipStream_t stream = (hipStream_t)p->args.hipModuleLaunchKernel.stream;
                Extrae_hipModuleLaunchKernel_Enter(f, gridDim, blockDim, sharedMemBytes, stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipModuleLaunchKernel_Exit();
            }
            ret = 1;
        }
        break;
        case HIP_API_ID_hipLaunchKernel:
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                char* address = (char*)p->args.hipLaunchKernel.function_address;
                dim3 dimBlocks = (dim3)p->args.hipLaunchKernel.dimBlocks;
                dim3 numBlocks = (dim3)p->args.hipLaunchKernel.numBlocks;
                unsigned gridDim = dimBlocks.x * dimBlocks.y * dimBlocks.z;
                unsigned blockDim = numBlocks.x * numBlocks.y * numBlocks.z;
                size_t sharedMemBytes = (size_t)p->args.hipLaunchKernel.sharedMemBytes;
                hipStream_t stream = (hipStream_t)p->args.hipLaunchKernel.stream;
                Extrae_hipLaunchKernel_Enter(address, gridDim, blockDim, sharedMemBytes, stream);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                Extrae_hipLaunchKernel_Exit();
            }
            ret = 1;
        }
        break;
    }
    return ret;  
}

#define Extrae_HIP_updateDepth(p) Extrae_HIP_updateDepth_((p->phase == ACTIVITY_API_PHASE_ENTER)?1:((p->phase == ACTIVITY_API_PHASE_EXIT)?-1:0))

static void
Extrae_ROCTRACER_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{

	UNREFERENCED_PARAMETER(domain);
    UNREFERENCED_PARAMETER(callback_data);

    if (IS_CID_FILTERED(cid)) return; 

	int ret = 0;

    if (!mpitrace_on || !Extrae_get_trace_HIP() || callback_data == NULL) 
        return;

    const hip_api_data_t* p = (const hip_api_data_t*) callback_data;
    const char* name = roctracer_op_string(domain, cid, 0);

	if((p->phase == ACTIVITY_API_PHASE_ENTER))
		Extrae_HIP_updateDepth(p);

	if((Extrae_HIP_getDepth() <= 1))
	{
        switch (domain)
        {
            case ACTIVITY_DOMAIN_HIP_API:
                ret = Extrae_HIP_API_callback(p, cid, callback_data, arg);
                break;
            default:
                #if LOG_OTHER_DOMAIN_UNTRACKED_CALLBACKS
                    fprintf(stderr, "Untracked Domain HIP event (domain = %d cid = %d  name = %s)\n", domain, cid, name);
                    fflush(stderr);
                    fprintf(stderr, "Untracked HIP ret = %d\n", ret);
                    fflush(stderr);
                #endif
                ret = -1;
        }
        if (ret == 0)
        {
            if (p->phase == ACTIVITY_API_PHASE_ENTER)
            {
                #if LOG_UNTRACKED_CALLBACKS
                    fprintf(stderr, "%s HIP call enter (domain = %d cid = %d)\n", name, domain, cid);
                    fflush(stderr);
                #endif
                TRACE_EVENT(LAST_READ_TIME, HIP_UNTRACKED_EV, cid);
            }
            else if (p->phase == ACTIVITY_API_PHASE_EXIT)
            {
                #if LOG_UNTRACKED_CALLBACKS
                    fprintf(stderr, "%s HIP call exit (domain = %d cid = %d)\n", name, domain, cid);
                    fflush(stderr);
                #endif
                TRACE_EVENT(TIME, HIP_UNTRACKED_EV, EVT_END);
            }
        }
    }
	if((p->phase == ACTIVITY_API_PHASE_EXIT))
		Extrae_HIP_updateDepth(p);
}

void Extrae_HIP_init (int rank)
{
    UNREFERENCED_PARAMETER(rank);
    Extrae_HIP_API_disable_calls();
    roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, Extrae_ROCTRACER_callback, NULL);
}
