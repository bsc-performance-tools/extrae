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

#include "cuda_common.h"

static void CUPTIAPI Extrae_CUPTI_callback (void *udata, CUpti_CallbackDomain domain,
	CUpti_CallbackId cbid, const CUpti_CallbackData *cbinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_CUDA())
		return;

	UNREFERENCED_PARAMETER(udata);

	/* We process only CUDA runtime calls */
	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{

		/* Check which event we have been subscribed. If we find a match through the switch,
		   we will call the hooks within the cuda_common.c providing the parameters from
		   the callback info parameter cbinfo->functionParams. The parameters are specific
		   to the routine that has been invoked. */
		switch (cbid)
		{
			case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
			{
			cudaLaunch_v3020_params *p = (cudaLaunch_v3020_params*) cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
#if CUPTI_API_VERSION >= 3
				Extrae_cudaLaunch_Enter (p->func);
#else
				Extrae_cudaLaunch_Enter (p->entry);
#endif
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
			{
			cudaConfigureCall_v3020_params *p = (cudaConfigureCall_v3020_params*) cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaConfigureCall_Enter (p->gridDim, p->blockDim, p->sharedMem, p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaConfigureCall_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020:
			{
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaThreadSynchronize_Enter ();
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaThreadSynchronize_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
			{
			cudaStreamCreate_v3020_params *p = (cudaStreamCreate_v3020_params*)cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamCreate_Enter (p->pStream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000:
			{
			cudaStreamCreateWithFlags_v5000_params *p = (cudaStreamCreateWithFlags_v5000_params*)cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamCreate_Enter (p->pStream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050:
			{
			cudaStreamCreateWithPriority_v5050_params *p = (cudaStreamCreateWithPriority_v5050_params*)cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamCreate_Enter (p->pStream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
			case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
			{
				cudaStreamSynchronize_v3020_params *p = (cudaStreamSynchronize_v3020_params *)cbinfo->functionParams;
				if (p != NULL)
				{
					if (cbinfo->callbackSite == CUPTI_API_ENTER)
						Extrae_cudaStreamSynchronize_Enter (p->stream);
					else if (cbinfo->callbackSite == CUPTI_API_EXIT)
						Extrae_cudaStreamSynchronize_Exit ();
				}
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
			{
			cudaMemcpy_v3020_params *p = (cudaMemcpy_v3020_params *)cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaMemcpy_Enter (p->dst, p->src, p->count, p->kind);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaMemcpy_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
			{
			cudaMemcpyAsync_v3020_params *p = (cudaMemcpyAsync_v3020_params*) cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaMemcpyAsync_Enter (p->dst, p->src, p->count, p->kind, p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaMemcpyAsync_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020:
			{
				if (cbinfo->callbackSite == CUPTI_API_EXIT)
					Extrae_cudaDeviceReset_Exit();
				else
					Extrae_cudaDeviceReset_Enter();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020:
			{
				if (cbinfo->callbackSite == CUPTI_API_EXIT)
					Extrae_cudaThreadExit_Exit();
				else
					Extrae_cudaThreadExit_Enter();
			}
			break;
		}
	}
}

void Extrae_CUDA_init (int rank)
{
	CUpti_SubscriberHandle subscriber;

	UNREFERENCED_PARAMETER(rank);

	/* Create a subscriber. All the routines will be handled at Extrae_CUPTI_callback */
	cuptiSubscribe (&subscriber, (CUpti_CallbackFunc) Extrae_CUPTI_callback, NULL);

	/* Activate callbacks for the following API calls:
	  CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020
          CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000
          CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050
	  CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020
	  CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020 */
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithPriority_v5050);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API,
		CUPTI_RUNTIME_TRACE_CBID_cudaThreadExit_v3020);
}

