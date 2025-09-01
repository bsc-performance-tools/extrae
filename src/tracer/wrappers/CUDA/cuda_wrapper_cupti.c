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

#include <cupti.h>
#include "cuda_common.h"
#include "cuda_probe.h"

#define LOG_UNTRACKED_CALLBACKS 0
#define LOG_OTHER_DOMAIN_UNTRACKED_CALLBACKS 0

#include "cuda_wrapper_cupti.h"

#if LOG_UNTRACKED_CALLBACKS
static CUptiResult (*cuptiGetCallbackName_real)(CUpti_CallbackDomain, uint32_t, const char**) = NULL;
#endif

/*
 * Values for CUPTI_API_VERSION.
 * Retrieved from cuda-11.5/extras/CUPTI/include/cupti_version.h
 *
 * v1 : CUDAToolsSDK 4.0
 * v2 : CUDAToolsSDK 4.1
 * v3 : CUDA Toolkit 5.0
 * v4 : CUDA Toolkit 5.5
 * v5 : CUDA Toolkit 6.0
 * v6 : CUDA Toolkit 6.5
 * v7 : CUDA Toolkit 6.5(with sm_52 support)
 * v8 : CUDA Toolkit 7.0
 * v9 : CUDA Toolkit 8.0
 * v10 : CUDA Toolkit 9.0
 * v11 : CUDA Toolkit 9.1
 * v12 : CUDA Toolkit 10.0, 10.1 and 10.2
 * v13 : CUDA Toolkit 11.0
 * v14 : CUDA Toolkit 11.1
 * v15 : CUDA Toolkit 11.2
 * v16 : CUDA Toolkit 11.5
 */

/*
 * Check which event we have been subscribed. If we find a match through the
 * switch, we will call the hooks within the cuda_common.c providing the
 * parameters from the callback info parameter cbinfo->functionParams. The
 * parameters are specific to the routine that has been invoked.
 */
static int
Extrae_DriverAPI_callback(CUpti_CallbackId cbid, const CUpti_CallbackData *cbinfo)
{
	int ret;

	switch (cbid)
	{
		case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
		{
			cuMemHostAlloc_params *p =
			  (cuMemHostAlloc_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaHostAlloc_Enter(p->pp, p->bytesize);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaHostAlloc_Exit();
			}
			ret = 1;
		}
		break;

		case CUPTI_DRIVER_TRACE_CBID_cuStreamCreate:
		{
			cuStreamCreate_params *p =
			  (cuStreamCreate_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaStreamCreate_Enter(p->phStream);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaStreamCreate_Exit();
			}
			ret = 1;
		}
		break;

		case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize:
		{
			cuStreamSynchronize_params *p =
			  (cuStreamSynchronize_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaStreamSynchronize_Enter(p->hStream);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaStreamSynchronize_Exit();
			}
			ret = 1;
		}
		break;

		case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
		{
			cuLaunchKernel_params *p =
			  (cuLaunchKernel_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaLaunch_Enter(p->f, p->hStream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit();

			ret = 1;
		}
		break;
	}

	return ret;
}

static int
Extrae_RuntimeAPI_callback(CUpti_CallbackId cbid, const CUpti_CallbackData *cbinfo)
{
	int ret = 0;
	switch (cbid)
	{
		/* 8 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
		{
			cudaConfigureCall_v3020_params *p =
			  (cudaConfigureCall_v3020_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaConfigureCall_Enter(
				  p->gridDim, p->blockDim, p->sharedMem, p->stream
				  );
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaConfigureCall_Exit();

			ret = 1;
		}
		break;

		/* 13 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
		{
			cudaLaunch_v3020_params *p =
			  (cudaLaunch_v3020_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
# if CUPTI_API_VERSION >= 3
				Extrae_cudaLaunch_Enter(p->func, NULL);
# else
				Extrae_cudaLaunch_Enter(p->entry, NULL);
# endif /* CUPTI_API_VERSION >= 3 */
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit();

			ret = 1;
		}
		break;

		/* 20 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
		{
			cudaMalloc_v3020_params *p =
			  (cudaMalloc_v3020_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaMalloc_Enter(CUDAMALLOC_VAL, p->devPtr, p->size);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaMalloc_Exit(CUDAMALLOC_VAL);
			}
			ret = 1;
		}
		break;

		/* 21 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020:
		{
			cudaMallocPitch_v3020_params *p =
			  (cudaMallocPitch_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaMalloc_Enter(
				  CUDAMALLOCPITCH_VAL, p->devPtr, p->width * p->height
				  );
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaMalloc_Exit(CUDAMALLOCPITCH_VAL);
			}
			ret = 1;
		}
		break;

		/* 22 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
		{
			cudaFree_v3020_params *p =
			  (cudaFree_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaFree_Enter(CUDAFREE_VAL, p->devPtr);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaFree_Exit(CUDAFREE_VAL);
			}
			ret = 1;
		}
		break;

		/* 23 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020:
		{
			cudaMallocArray_v3020_params *p =
			  (cudaMallocArray_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaMalloc_Enter(
				  CUDAMALLOCARRAY_VAL, (void *)p->array, p->width * p->height
				  );
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaMalloc_Exit(CUDAMALLOCARRAY_VAL);
			}
			ret = 1;
		}
		break;

		/* 24 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020:
		{
			cudaFreeArray_v3020_params *p =
			  (cudaFreeArray_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaFree_Enter(CUDAFREEARRAY_VAL, (void *)p->array);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaFree_Exit(CUDAFREEARRAY_VAL);
			}
			ret = 1;
		}
		break;

		/* 25 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
		{
			cudaMallocHost_v3020_params *p =
			  (cudaMallocArray_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaMalloc_Enter(
				  CUDAMALLOCHOST_VAL, p->ptr, p->size
				  );
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaMalloc_Exit(CUDAMALLOCHOST_VAL);
			}
			ret = 1;
		}
		break;

		/* 26 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
		{
			cudaFreeHost_v3020_params *p =
			  (cudaFreeHost_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaFree_Enter(CUDAFREEHOST_VAL, p->ptr);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaFree_Exit(CUDAFREEHOST_VAL);
			}
			ret = 1;
		}
		break;

		/* 27 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020:
		{
			cudaHostAlloc_v3020_params *p =
			  (cudaHostAlloc_v3020_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaHostAlloc_Enter(p->pHost, p->size);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaHostAlloc_Exit();
			}
			ret = 1;
		}
		break;

		/* 31 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
		{
			cudaMemcpy_v3020_params *p =
			  (cudaMemcpy_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaMemcpy_Enter(p->dst, p->src, p->count, p->kind);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaMemcpy_Exit();

			ret = 1;
		}
		break;

		/* 41 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
		{
			cudaMemcpyAsync_v3020_params *p =
			  (cudaMemcpyAsync_v3020_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaMemcpyAsync_Enter(
				  p->dst, p->src, p->count, p->kind, p->stream
				  );
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaMemcpyAsync_Exit();

			ret = 1;
		}
		break;

		/* 49 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020:
		{
			cudaMemset_v3020_params *p =
			  (cudaMemset_v3020_params *)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaMemset_Enter(p->devPtr, p->count);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaMemset_Exit();

			ret = 1;
		}
		break;

		/* 123 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020:
		{
			if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaThreadExit_Exit();
			else
				Extrae_cudaThreadExit_Enter();

			ret = 1;
		}
		break;

		/* 126 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020:
		{
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaThreadSynchronize_Enter();
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaThreadSynchronize_Exit();

			ret = 1;
		}
		break;

		/* 129 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
		{
			cudaStreamCreate_v3020_params *p =
			  (cudaStreamCreate_v3020_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamCreate_Enter(p->pStream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit();

			ret = 1;
		}
		break;

		/* 130 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
		{
			cudaStreamDestroy_v3020_params *p =
			  (cudaStreamDestroy_v3020_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamDestroy_Enter (p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamDestroy_Exit();

			ret = 1;
		}
		break;

		/* 131 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
		{
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaDeviceSynchronize_Enter();
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaDeviceSynchronize_Exit();

			ret = 1;
		}
		break;

		/* 135 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020:
		{
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaEventRecord_Enter();
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaEventRecord_Exit();

			ret = 1;
		}
		break;

		/* 137 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020:
		{
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaEventSynchronize_Enter();
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaEventSynchronize_Exit();

			ret = 1;
		}
		break;

		/* 164 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020:
		{
			if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaDeviceReset_Exit();
			else
				Extrae_cudaDeviceReset_Enter();

			ret = 1;
		}
		break;

		/* 165 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
		{
			cudaStreamSynchronize_v3020_params *p =
			  (cudaStreamSynchronize_v3020_params *)cbinfo->functionParams;

			if (p != NULL)
			{
				if (cbinfo->callbackSite == CUPTI_API_ENTER)
					Extrae_cudaStreamSynchronize_Enter(p->stream);
				else if (cbinfo->callbackSite == CUPTI_API_EXIT)
					Extrae_cudaStreamSynchronize_Exit();
			}

			ret = 1;
		}
		break;

#if CUPTI_API_VERSION >= 3
		/* 198 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000:
		{
			cudaStreamCreateWithFlags_v5000_params *p =
			  (cudaStreamCreateWithFlags_v5000_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamCreate_Enter(p->pStream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit();

			ret = 1;
		}
		break;
#endif /* CUPTI_API_VERSION >= 3 */

#if CUPTI_API_VERSION >= 4
		/* 201 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v5050:
		{
			cudaStreamDestroy_v5050_params *p =
			  (cudaStreamDestroy_v5050_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamDestroy_Enter(p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamDestroy_Exit();

			ret = 1;
		}
		break;

		/* 202 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050:
		{
			cudaStreamCreateWithPriority_v5050_params *p =
			  (cudaStreamCreateWithPriority_v5050_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamCreate_Enter(p->pStream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit();

			ret = 1;
		}
		break;
#endif /* CUPTI_API_VERSION >= 4 */

#if CUPTI_API_VERSION >= 5
		/* 206 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
		{
			cudaMallocManaged_v6000_params *p =
			  (cudaMallocManaged_v6000_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
			{
				Extrae_cudaMalloc_Enter(CUDAMALLOC_VAL, p->devPtr, p->size);
			}
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
			{
				Extrae_cudaMalloc_Exit(CUDAMALLOC_VAL);
			}

			ret = 1;
		}
		break;
#endif /* CUPTI_API_VERSION >= 5 */

#if CUPTI_API_VERSION >= 8
		/* 211 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
		{
			cudaLaunchKernel_v7000_params *p =
			  (cudaLaunchKernel_v7000_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaLaunch_Enter(p->func, p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit();

			ret = 1;
		}
		break;

		/* 214 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000:
		{
			cudaLaunchKernel_ptsz_v7000_params *p =
			  (cudaLaunchKernel_ptsz_v7000_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaLaunch_Enter(p->func, p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit();

			ret = 1;
		}
		break;
#endif /* CUPTI_API_VERSION >= 8 */

#if CUPTI_API_VERSION >= 10
		/* 269 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_v9000:
		{
			cudaLaunchCooperativeKernel_v9000_params *p =
			  (cudaLaunchCooperativeKernel_v9000_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaLaunch_Enter(p->func, p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit();

			ret = 1;
		}
		break;

		/* 270 */
		case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchCooperativeKernel_ptsz_v9000:
		{
			cudaLaunchCooperativeKernel_ptsz_v9000_params *p =
			  (cudaLaunchCooperativeKernel_ptsz_v9000_params*)cbinfo->functionParams;

			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaLaunch_Enter(p->func, p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit();

			ret = 1;
		}
		break;
#endif /* CUPTI_API_VERSION >= 10 */
	}

	return ret;
}

#define Extrae_CUDA_updateDepth(cbinfo) Extrae_CUDA_updateDepth_((cbinfo->callbackSite == CUPTI_API_ENTER)?1:((cbinfo->callbackSite == CUPTI_API_EXIT)?-1:0))

/*
 * Process only CUDA Runtime and Driver API calls. Depending on which type the
 * call is, jump to the specific callback management function.
 */
static void
CUPTIAPI Extrae_CUPTI_callback(void *udata, CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid, const CUpti_CallbackData *cbinfo)
{
	UNREFERENCED_PARAMETER(udata);
	UNREFERENCED_PARAMETER(domain);
	int ret = 0;

	if( cbinfo != NULL )
	{
		if (cbinfo->callbackSite == CUPTI_API_ENTER) Backend_Enter_Instrumentation();
	}
	else
	{
		Backend_Enter_Instrumentation();
	}

	if (EXTRAE_ON() && Extrae_get_trace_CUDA())
	{
		if( Backend_Get_InstrumentationLevel() == 1)
		{
			switch (domain)
			{
				case CUPTI_CB_DOMAIN_DRIVER_API:
					ret = Extrae_DriverAPI_callback(cbid, cbinfo);
					break;
				case CUPTI_CB_DOMAIN_RUNTIME_API:
					ret = Extrae_RuntimeAPI_callback(cbid, cbinfo);
					break;
				default:
					ret = -1;
		#if LOG_OTHER_DOMAIN_UNTRACKED_CALLBACKS
					fprintf(stderr, "CUDA call (domain = %d cbid = %d\n", domain, cbid);
		#endif
			}

			if (ret == 0)
			{
				if (cbinfo->callbackSite == CUPTI_API_ENTER)
				{
		#if LOG_UNTRACKED_CALLBACKS
					const char *callbackName = "Untracked";
					if (cuptiGetCallbackName_real != NULL)
					{
						cuptiGetCallbackName_real
							(domain, cbid, &callbackName);
					}
					fprintf(stderr, "%s CUDA call (domain = %d cbid = %d)\n", callbackName, domain, cbid);
		#endif

					TRACE_EVENT(LAST_READ_TIME, CUDA_UNTRACKED_EV, cbid);
				}
				else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				{
					TRACE_EVENT(TIME, CUDA_UNTRACKED_EV, EVT_END);
				}
			}
		}
	}
	if( cbinfo != NULL )
	{
		if((cbinfo->callbackSite == CUPTI_API_EXIT)) Backend_Leave_Instrumentation();
	}
	else
	{
		Backend_Leave_Instrumentation();
	}
}

void Extrae_CUDA_init (int rank)
{
	CUpti_SubscriberHandle subscriber;

	UNREFERENCED_PARAMETER(rank);

#if LOG_UNTRACKED_CALLBACKS
cuptiGetCallbackName_real = (CUptiResult(*)(CUpti_CallbackDomain, uint32_t, const char**)) dlsym(RTLD_NEXT, "cuptiGetCallbackName");
#endif

	/* Create a subscriber. All the routines will be handled at Extrae_CUPTI_callback */
	cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) Extrae_CUPTI_callback, NULL);

	/* Enable all callbacks in all domains */
	cuptiEnableAllDomains(1, subscriber);

	/* Disable uninteresting callbacks */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020); /* 1 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaRuntimeGetVersion_v3020); /* 2 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceCount_v3020); /* 3 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020); /* 4 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaChooseDevice_v3020); /* 5 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetChannelDesc_v3020); /* 6 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaCreateChannelDesc_v3020); /* 7 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020); /* 9 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetLastError_v3020); /* 10 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaPeekAtLastError_v3020); /* 11 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetErrorString_v3020); /* 12 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFuncSetCacheConfig_v3020); /* 14 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFuncGetAttributes_v3020); /* 15 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020); /* 16 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetDevice_v3020); /* 17 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaSetValidDevices_v3020); /* 18 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaSetDeviceFlags_v3020); /* 19 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaHostGetDevicePointer_v3020); /* 28 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaHostGetFlags_v3020); /* 29 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemGetInfo_v3020); /* 30 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolAddress_v3020); /* 53 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetSymbolSize_v3020); /* 54 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureAlignmentOffset_v3020); /* 59 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetTextureReference_v3020); /* 60 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetSurfaceReference_v3020); /* 62 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGLSetGLDevice_v3020); /* 63 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGLSetBufferObjectMapFlags_v3020); /* 68 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaWGLGetDevice_v3020); /* 71 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaEventCreateWithFlags_v3020); /* 134 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020); /* 136 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaEventElapsedTime_v3020); /* 139 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaHostRegister_v4000); /* 152 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSetCacheConfig_v3020); /* 169 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000); /* 171 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000); /* 172 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaDeviceGetAttribute_v5000); /* 200 */
	cuptiEnableCallback(0, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceFlags_v7000); /* 212 */
}
