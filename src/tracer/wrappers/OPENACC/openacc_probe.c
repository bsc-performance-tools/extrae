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

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "openacc_probe.h"

void
Probe_OPENACC_device_init_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, EVT_BEGIN, OPENACC_INIT_VAL);
}

void
Probe_OPENACC_device_init_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, OPENACC_INIT_VAL);
}

void
Probe_OPENACC_device_shutdown_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, EVT_BEGIN, OPENACC_SHUTDOWN_VAL);
}

void
Probe_OPENACC_device_shutdown_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, OPENACC_SHUTDOWN_VAL);
}

void
Probe_OPENACC_enter_data_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, EVT_BEGIN, OPENACC_ENTER_DATA_VAL);
}

void
Probe_OPENACC_enter_data_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, OPENACC_ENTER_DATA_VAL);
}

void
Probe_OPENACC_exit_data_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, EVT_BEGIN, OPENACC_EXIT_DATA_VAL);
}

void
Probe_OPENACC_exit_data_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, OPENACC_EXIT_DATA_VAL);
}

void
Probe_OPENACC_create(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_CREATE_VAL, EMPTY);
}

void
Probe_OPENACC_delete(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_DELETE_VAL, EMPTY);
}

void
Probe_OPENACC_alloc(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_ALLOC_VAL, EMPTY);
}

void
Probe_OPENACC_free(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_FREE_VAL, EMPTY);
}

void
Probe_OPENACC_update_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, EVT_BEGIN, OPENACC_UPDATE_VAL);
}

void
Probe_OPENACC_update_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, OPENACC_UPDATE_VAL);
}

void
Probe_OPENACC_compute_construct_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, EVT_BEGIN, OPENACC_COMPUTE_VAL);
}

void
Probe_OPENACC_compute_construct_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, OPENACC_COMPUTE_VAL);
}

void
Probe_OPENACC_enqueue_launch_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_LAUNCH_EV, EVT_BEGIN, OPENACC_ENQUEUE_LAUNCH_VAL);
}

void
Probe_OPENACC_enqueue_launch_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_LAUNCH_EV, EVT_END, OPENACC_ENQUEUE_LAUNCH_VAL);
}

void
Probe_OPENACC_enqueue_upload_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, EVT_BEGIN, OPENACC_ENQUEUE_UPLOAD_VAL);
}

void
Probe_OPENACC_enqueue_upload_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_DATA_EV, EVT_END, OPENACC_ENQUEUE_UPLOAD_VAL);
}

void
Probe_OPENACC_enqueue_download_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, EVT_BEGIN, OPENACC_ENQUEUE_DOWNLOAD_VAL);
}

void
Probe_OPENACC_enqueue_download_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_DATA_EV, EVT_END, OPENACC_ENQUEUE_DOWNLOAD_VAL);
}

void
Probe_OPENACC_wait_start(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, EVT_BEGIN, OPENACC_WAIT_VAL);
}

void
Probe_OPENACC_wait_end(int implicit)
{
	if (mpitrace_on && !implicit)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, OPENACC_WAIT_VAL);
}
