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
	int devid;

	if (!mpitrace_on)
		return;

	UNREFERENCED_PARAMETER(udata);

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	if (domain == CUPTI_CB_DOMAIN_RUNTIME_API)
	{
		switch (cbid)
		{
			case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaLaunch_Enter (devid, (cudaLaunch_v3020_params*)cbinfo->functionParams);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaLaunch_Exit (devid, (cudaLaunch_v3020_params*)cbinfo->functionParams);
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaConfigureCall_Enter (devid, (cudaConfigureCall_v3020_params*)cbinfo->functionParams);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaConfigureCall_Exit (devid, (cudaConfigureCall_v3020_params*)cbinfo->functionParams);
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020:
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaThreadSynchronize_Enter (devid);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaThreadSynchronize_Exit (devid);
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
			if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamCreate_Exit (devid, (cudaStreamCreate_v3020_params*)cbinfo->functionParams);
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaStreamSynchronize_Enter (devid, (cudaStreamSynchronize_v3020_params*)cbinfo->functionParams);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaStreamSynchronize_Exit (devid, (cudaStreamSynchronize_v3020_params*)cbinfo->functionParams);
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaMemcpy_Enter (devid, (cudaMemcpy_v3020_params*)cbinfo->functionParams);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaMemcpy_Exit (devid, (cudaMemcpy_v3020_params*)cbinfo->functionParams);
			break;

			case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
			if (cbinfo->callbackSite == CUPTI_API_ENTER)
				Extrae_cudaMemcpyAsync_Enter (devid, (cudaMemcpyAsync_v3020_params*)cbinfo->functionParams);
			else if (cbinfo->callbackSite == CUPTI_API_EXIT)
				Extrae_cudaMemcpyAsync_Exit (devid, (cudaMemcpyAsync_v3020_params*)cbinfo->functionParams);
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

