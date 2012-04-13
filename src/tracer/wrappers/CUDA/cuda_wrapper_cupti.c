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

#include "wrapper.h"

#include "cuda_common.h"

static void CUPTIAPI Extrae_CUPTI_callback (void *udata, CUpti_CallbackDomain domain,
	CUpti_CallbackId cbid, const CUpti_CallbackData *cbinfo)
{
	if (!mpitrace_on)
		return;

	UNREFERENCED_PARAMETER(udata);

	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{
		switch (cbid)
		{
			case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
			{
			cudaLaunch_v3020_params *p = (cudaLaunch_v3020_params*) cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaLaunch_Enter (p->entry);
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
			if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit ();
			}
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
			{
			cudaStreamSynchronize_v3020_params *p = (cudaStreamSynchronize_v3020_params *)cbinfo->functionParams;
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamSynchronize_Enter (p->stream);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamSynchronize_Exit ();
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
		}
	}
}

void Extrae_CUDA_init (int rank)
{
	CUpti_SubscriberHandle subscriber;

	UNREFERENCED_PARAMETER(rank);

	cuptiSubscribe (&subscriber, (CUpti_CallbackFunc) Extrae_CUPTI_callback, NULL);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);
	cuptiEnableCallback (1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020);
}

