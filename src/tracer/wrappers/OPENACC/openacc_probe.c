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
Probe_OPENACC_device_init_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_INIT_VAL, EMPTY);
}

void
Probe_OPENACC_device_init_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_device_shutdown_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_SHUTDOWN_VAL, EMPTY);
}

void
Probe_OPENACC_device_shutdown_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_enter_data_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_ENTER_DATA_VAL, EMPTY);
}

void
Probe_OPENACC_enter_data_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_exit_data_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_EXIT_DATA_VAL, EMPTY);
}

void
Probe_OPENACC_exit_data_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_create()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_CREATE_VAL, EMPTY);
}

void
Probe_OPENACC_delete()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_DELETE_VAL, EMPTY);
}

void
Probe_OPENACC_alloc()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_ALLOC_VAL, EMPTY);
}

void
Probe_OPENACC_free()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_DATA_EV, OPENACC_FREE_VAL, EMPTY);
}

void
Probe_OPENACC_update_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_UPDATE_VAL, EMPTY);
}

void
Probe_OPENACC_update_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_compute_construct_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_COMPUTE_VAL, EMPTY);
}

void
Probe_OPENACC_compute_construct_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_enqueue_launch_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_ENQUEUE_KERNEL_LAUNCH_VAL, EMPTY);
}

void
Probe_OPENACC_enqueue_launch_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_enqueue_upload_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_ENQUEUE_UPLOAD_VAL, EMPTY);
}

void
Probe_OPENACC_enqueue_upload_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_enqueue_download_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_ENQUEUE_DOWNLOAD_VAL, EMPTY);
}

void
Probe_OPENACC_enqueue_download_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}

void
Probe_OPENACC_wait_start()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, OPENACC_EV, OPENACC_WAIT_VAL, EMPTY);
}

void
Probe_OPENACC_wait_end()
{
	if (mpitrace_on)
		TRACE_MISCEVENTANDCOUNTERS(TIME, OPENACC_EV, EVT_END, EMPTY);
}
