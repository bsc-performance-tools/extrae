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

#include "wrapper.h"

#include "trace_gaspi_macros.h"
#include "gaspi_probe.h"

static int trace_gaspi = TRUE;
static int trace_gaspi_hwc = TRUE;

void
Extrae_set_trace_GASPI(int trace)
{
	trace_gaspi = trace;
}

int
Extrae_get_trace_GASPI()
{
	return trace_gaspi;
}

void
Extrae_set_trace_GASPI_HWC(int trace)
{
	trace_gaspi_hwc = trace;
}

int
Extrae_get_trace_GASPI_HWC()
{
	return trace_gaspi_hwc;
}

void
Probe_GASPI_init_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_INIT_EV, EVT_BEGIN, EMPTY, Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_init_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_INIT_EV, EVT_END, EMPTY, Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_term_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_TERM_VAL, Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_term_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_TERM_VAL, Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_connect_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_CONNECT_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_connect_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_CONNECT_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_disconnect_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_DISCONNECT_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_disconnect_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_DISCONNECT_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_create_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_GROUP_CREATE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_create_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_GROUP_CREATE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_add_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_GROUP_ADD_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_add_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_GROUP_ADD_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_commit_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_GROUP_COMMIT_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_commit_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_GROUP_COMMIT_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_delete_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_GROUP_DELETE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_group_delete_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_GROUP_DELETE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_segment_alloc_Entry(const gaspi_size_t size)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_SEGMENT_ALLOC_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
	}
}

void
Probe_GASPI_segment_alloc_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_SEGMENT_ALLOC_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_segment_register_Entry(const gaspi_rank_t rank)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_SEGMENT_REGISTER_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
	}
}

void
Probe_GASPI_segment_register_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_SEGMENT_REGISTER_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_segment_create_Entry(const gaspi_size_t size)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(
		    LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_SEGMENT_CREATE_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
	}
}

void
Probe_GASPI_segment_create_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_SEGMENT_CREATE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_segment_bind_Entry(const gaspi_size_t size)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_SEGMENT_BIND_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
	}
}

void
Probe_GASPI_segment_bind_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_SEGMENT_BIND_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_segment_use_Entry(const gaspi_size_t size)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_SEGMENT_USE_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
	}
}

void
Probe_GASPI_segment_use_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_SEGMENT_USE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_segment_delete_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_SEGMENT_DELETE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_segment_delete_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_SEGMENT_DELETE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_write_Entry(const gaspi_rank_t rank, const gaspi_size_t size,
    const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_WRITE_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_write_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_WRITE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_read_Entry(const gaspi_rank_t rank, const gaspi_size_t size,
    const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_READ_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_read_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_READ_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_wait_Entry(const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_WAIT_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_wait_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_WAIT_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_notify_Entry(const gaspi_rank_t rank,
    const gaspi_notification_id_t notification_id, const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_NOTIFICATION_ID_EV,
		    (UINT64)notification_id);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_notify_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_notify_waitsome_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_NOTIFY_WAITSOME_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_notify_waitsome_Exit(const gaspi_notification_id_t first_id)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_NOTIFY_WAITSOME_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_NOTIFICATION_ID_EV,
		    (UINT64)first_id);
	}
}

void
Probe_GASPI_notify_reset_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_NOTIFY_RESET_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_notify_reset_Exit(
    const gaspi_notification_id_t old_notification_id)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_NOTIFY_RESET_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_NOTIFICATION_ID_EV,
		    (UINT64)old_notification_id);
	}
}

void
Probe_GASPI_write_notify_Entry(const gaspi_rank_t rank, const gaspi_size_t size,
    const gaspi_notification_id_t notification_id, const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_WRITE_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_NOTIFICATION_ID_EV, (UINT64)notification_id);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_write_notify_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_WRITE_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_write_list_Entry(const gaspi_rank_t rank, gaspi_size_t * const size,
    const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_WRITE_LIST_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_write_list_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_WRITE_LIST_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_write_list_notify_Entry(const gaspi_rank_t rank,
    gaspi_size_t * const size, const gaspi_notification_id_t notification_id,
    const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_WRITE_LIST_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_NOTIFICATION_ID_EV, (UINT64)notification_id);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_write_list_notify_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_WRITE_LIST_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_read_notify_Entry(const gaspi_rank_t rank, const gaspi_size_t size,
    const gaspi_notification_id_t notification_id,
    const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_READ_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_NOTIFICATION_ID_EV, (UINT64)notification_id);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_read_notify_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_READ_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_read_list_Entry(const gaspi_rank_t rank, gaspi_size_t * const size,
    const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_READ_LIST_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_read_list_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_READ_LIST_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_read_list_notify_Entry(const gaspi_rank_t rank,
    gaspi_size_t * const size, const gaspi_notification_id_t notification_id,
    const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_READ_LIST_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_NOTIFICATION_ID_EV, (UINT64)notification_id);
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_read_list_notify_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_READ_LIST_NOTIFY_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_passive_send_Entry(const gaspi_rank_t rank, const gaspi_size_t size)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_PASSIVE_SEND_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
	}
}

void
Probe_GASPI_passive_send_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_PASSIVE_SEND_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_passive_receive_Entry(const gaspi_size_t size)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_PASSIVE_RECEIVE_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GASPI_SIZE_EV, (UINT64)size);
	}
}

void
Probe_GASPI_passive_receive_Exit(const gaspi_rank_t rem_rank)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_PASSIVE_RECEIVE_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rem_rank);
	}
}

void
Probe_GASPI_atomic_fetch_add_Entry(const gaspi_rank_t rank)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_ATOMIC_FETCH_ADD_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
	}
}

void
Probe_GASPI_atomic_fetch_add_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_ATOMIC_FETCH_ADD_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_atomic_compare_swap_Entry(const gaspi_rank_t rank)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_ATOMIC_COMPARE_SWAP_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_RANK_EV, (UINT64)rank);
	}
}

void
Probe_GASPI_atomic_compare_swap_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_ATOMIC_COMPARE_SWAP_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_barrier_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_BARRIER_VAL, Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_barrier_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_BARRIER_VAL, Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_allreduce_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_ALLREDUCE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_allreduce_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_ALLREDUCE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_allreduce_user_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_ALLREDUCE_USER_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_allreduce_user_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_ALLREDUCE_USER_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_queue_create_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_QUEUE_CREATE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}

void
Probe_GASPI_queue_create_Exit(const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_QUEUE_CREATE_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_queue_delete_Entry(const gaspi_queue_id_t queue)
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(LAST_READ_TIME, GASPI_EV, EVT_BEGIN, GASPI_QUEUE_DELETE_VAL,
		    Extrae_get_trace_GASPI_HWC());
		TRACE_GASPI_PARAM(LAST_READ_TIME, GASPI_QUEUE_ID_EV, (UINT64)queue);
	}
}

void
Probe_GASPI_queue_delete_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GASPI())
	{
		TRACE_GASPI_EVENTANDCOUNTERS(TIME, GASPI_EV, EVT_END, GASPI_QUEUE_DELETE_VAL,
		    Extrae_get_trace_GASPI_HWC());
	}
}
