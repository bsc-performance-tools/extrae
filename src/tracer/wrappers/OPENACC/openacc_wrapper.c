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
#include <acc_prof.h>
#include "wrapper.h"
#if defined(CUDA_SUPPORT)
# warning "Including cuda_common.h"
# include "cuda_common.h"
#endif
#include "openacc_probe.h"

static int trace_openacc = TRUE;

void
Extrae_set_trace_OpenACC(int b)
{
	trace_openacc = b;
}

int
Extrae_get_trace_OpenACC()
{
	return trace_openacc;
}

void
OACC_init(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC())
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
	else
#endif

	Backend_Enter_Instrumentation ();

	acc_other_event_info ei = eventinfo->other_event;
	switch(ei.event_type)
	{
		case acc_ev_device_init_start:
			Probe_OPENACC_device_init_start(ei.implicit);
			break;
		case acc_ev_device_init_end:
			Probe_OPENACC_device_init_end(ei.implicit);
			break;
		case acc_ev_device_shutdown_start:
			Probe_OPENACC_device_shutdown_start(ei.implicit);
			break;
		case acc_ev_device_shutdown_end:
			Probe_OPENACC_device_shutdown_end(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void
OACC_data(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC() || apiinfo->device_api != acc_device_api_cuda)
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
	else
#endif

	Backend_Enter_Instrumentation ();

	acc_other_event_info ei = eventinfo->other_event;
	switch(ei.event_type)
	{
		case acc_ev_enter_data_start:
			Probe_OPENACC_enter_data_start(ei.implicit);
			break;
		case acc_ev_enter_data_end:
			Probe_OPENACC_enter_data_end(ei.implicit);
			break;
		case acc_ev_exit_data_start:
			Probe_OPENACC_exit_data_start(ei.implicit);
			break;
		case acc_ev_exit_data_end:
			Probe_OPENACC_exit_data_end(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void
OACC_data_alloc(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC())
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
	else
#endif

	Backend_Enter_Instrumentation ();

	acc_data_event_info ei = eventinfo->data_event;
	switch(ei.event_type)
	{
		case acc_ev_create:
			Probe_OPENACC_create(ei.implicit);
			break;
		case acc_ev_delete:
			Probe_OPENACC_delete(ei.implicit);
			break;
		case acc_ev_alloc:
			Probe_OPENACC_alloc(ei.implicit);
			break;
		case acc_ev_free:
			Probe_OPENACC_free(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void
OACC_update(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC())
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
#endif

	Backend_Enter_Instrumentation ();

	acc_other_event_info ei = eventinfo->other_event;
	switch (ei.event_type)
	{
		case acc_ev_update_start:
			Probe_OPENACC_update_start(ei.implicit);
			break;
		case acc_ev_update_end:
			Probe_OPENACC_update_end(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void
OACC_compute(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC())
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
	else
#endif

	Backend_Enter_Instrumentation ();

	acc_other_event_info ei = eventinfo->other_event;
	switch (ei.event_type)
	{
		case acc_ev_compute_construct_start:
			Probe_OPENACC_compute_construct_start(ei.implicit);
			break;
		case acc_ev_compute_construct_end:
			Probe_OPENACC_compute_construct_end(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void
OACC_launch(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC())
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
	else
#endif

	Backend_Enter_Instrumentation ();

	acc_launch_event_info ei = eventinfo->launch_event;
	switch(ei.event_type)
	{
		case acc_ev_enqueue_launch_start:
			Probe_OPENACC_enqueue_launch_start(ei.implicit);
			break;
		case acc_ev_enqueue_launch_end:
			Probe_OPENACC_enqueue_launch_end(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void
OACC_data_update(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC())
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
	else
#endif

	Backend_Enter_Instrumentation ();

	acc_data_event_info ei = eventinfo->data_event;
	switch(ei.event_type)
	{
		case acc_ev_enqueue_upload_start:
			Probe_OPENACC_enqueue_upload_start(ei.implicit);
			break;
		case acc_ev_enqueue_upload_end:
			Probe_OPENACC_enqueue_upload_end(ei.implicit);
			break;
		case acc_ev_enqueue_download_start:
			Probe_OPENACC_enqueue_download_start(ei.implicit);
			break;
		case acc_ev_enqueue_download_end:
			Probe_OPENACC_enqueue_download_end(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void OACC_wait(acc_prof_info *profinfo, acc_event_info *eventinfo, acc_api_info *apiinfo)
{
	if (!mpitrace_on || !Extrae_get_trace_OpenACC())
		return;

#if defined(CUDA_SUPPORT)
	if (apiinfo->device_api == acc_device_api_cuda)
	{
		Extrae_CUDA_Initialize(profinfo->device_number);
	}
	else
#endif

	Backend_Enter_Instrumentation ();

	acc_other_event_info ei = eventinfo->other_event;
	switch(ei.event_type)
	{
		case acc_ev_wait_start:
			Probe_OPENACC_wait_start(ei.implicit);
			break;
		case acc_ev_wait_end:
			Probe_OPENACC_wait_end(ei.implicit);
			break;
		default:
			break;
	}

	Backend_Leave_Instrumentation ();
}

void
Extrae_OACC_init(int rank)
{
	// Device Initialization and Shutdown (acc_other_event_info)
	acc_prof_register(acc_ev_device_init_start, OACC_init, 0);
	acc_prof_register(acc_ev_device_init_end, OACC_init, 0);
	acc_prof_register(acc_ev_device_shutdown_start, OACC_init, 0);
	acc_prof_register(acc_ev_device_shutdown_end, OACC_init, 0);
	// Enter Data and Exit Data (acc_other_event_info)
	acc_prof_register(acc_ev_enter_data_start, OACC_data, 0);
	acc_prof_register(acc_ev_enter_data_end, OACC_data, 0);
	acc_prof_register(acc_ev_exit_data_start, OACC_data, 0);
	acc_prof_register(acc_ev_exit_data_end, OACC_data, 0);
	// Data Allocation (acc_data_event_info)
	acc_prof_register(acc_ev_create, OACC_data_alloc, 0);
	acc_prof_register(acc_ev_delete, OACC_data_alloc, 0);
	acc_prof_register(acc_ev_alloc, OACC_data_alloc, 0);
	acc_prof_register(acc_ev_free, OACC_data_alloc, 0);
	// Update Directive (acc_other_event_info)
	acc_prof_register(acc_ev_update_start, OACC_update, 0);
	acc_prof_register(acc_ev_update_end, OACC_update, 0);
	// Compute Construct (acc_other_event_info)
	acc_prof_register(acc_ev_compute_construct_start, OACC_compute, 0);
	acc_prof_register(acc_ev_compute_construct_end, OACC_compute, 0);
	// Enqueue Kernel Launch (acc_launch_event_info)
	acc_prof_register(acc_ev_enqueue_launch_start, OACC_launch, 0);
	acc_prof_register(acc_ev_enqueue_launch_end, OACC_launch, 0);
	// Enqueue Data Update (acc_data_event_info)
	acc_prof_register(acc_ev_enqueue_upload_start, OACC_data_update, 0);
	acc_prof_register(acc_ev_enqueue_upload_end, OACC_data_update, 0);
	acc_prof_register(acc_ev_enqueue_download_start, OACC_data_update, 0);
	acc_prof_register(acc_ev_enqueue_download_end, OACC_data_update, 0);
	// Wait (acc_other_event_info)
	acc_prof_register(acc_ev_wait_start, OACC_wait, 0);
	acc_prof_register(acc_ev_wait_end, OACC_wait, 0);
}
