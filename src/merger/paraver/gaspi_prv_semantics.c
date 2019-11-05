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

#include "events.h"
#include "file_set.h"
#include "paraver_generator.h"
#include "paraver_state.h"
#include "record.h"
#include "semantics.h"

#include "gaspi_prv_events.h"

static int
GASPI_Event(event_t *current_event, unsigned long long current_time, unsigned cpu,
    unsigned ptask, unsigned task, unsigned thread, FileSet_t *fset)
{
	unsigned int EvType, nEvType;
	unsigned long long EvValue, nEvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent(current_event);
	EvValue = Get_EvValue(current_event);

	switch (EvType)
	{
		case GASPI_INIT_EV:
		case GASPI_TERM_EV:
			Switch_State(STATE_INITFINI, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_BARRIER_EV:
			Switch_State(STATE_BARRIER, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_WRITE_EV:
		case GASPI_READ_EV:
		case GASPI_WAIT_EV:
		case GASPI_NOTIFY_EV:
		case GASPI_NOTIFY_WAITSOME_EV:
		case GASPI_NOTIFY_RESET_EV:
		case GASPI_WRITE_NOTIFY_EV:
		case GASPI_WRITE_LIST_EV:
		case GASPI_WRITE_LIST_NOTIFY_EV:
		case GASPI_READ_LIST_EV:
			Switch_State(STATE_1SIDED, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_CONNECT_EV:
		case GASPI_DISCONNECT_EV:
			Switch_State(STATE_SYNC, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_ALLREDUCE_EV:
		case GASPI_ALLREDUCE_USER_EV:
			Switch_State(STATE_BCAST, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_GROUP_CREATE_EV:
		case GASPI_GROUP_ADD_EV:
		case GASPI_GROUP_COMMIT_EV:
		case GASPI_GROUP_DELETE_EV:
			Switch_State(STATE_MIXED, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_SEGMENT_ALLOC_EV:
		case GASPI_SEGMENT_REGISTER_EV:
		case GASPI_SEGMENT_CREATE_EV:
		case GASPI_SEGMENT_BIND_EV:
		case GASPI_SEGMENT_USE_EV:
		case GASPI_SEGMENT_DELETE_EV:
			Switch_State(STATE_ALLOCMEM, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_PASSIVE_SEND_EV:
			Switch_State(STATE_SEND, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_PASSIVE_RECEIVE_EV:
			Switch_State(STATE_WAITMESS, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GASPI_ATOMIC_FETCH_ADD_EV:
		case GASPI_ATOMIC_COMPARE_SWAP_EV:
			Switch_State(STATE_ATOMIC_MEM_OP, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
	}

	Translate_GASPI_Operation(EvType, EvValue, &nEvType, &nEvValue);
	trace_paraver_event(cpu, ptask, task, thread, current_time, nEvType, nEvValue);

	return 0;
}

static int
GASPI_Param(event_t *current_event, unsigned long long current_time, unsigned cpu,
    unsigned ptask, unsigned task, unsigned thread, FileSet_t *fset)
{
	unsigned int EvType;
	unsigned long long EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent(current_event);
	EvValue = Get_EvValue(current_event);

	if (EvType == GASPI_RANK_EV)
	{
		EvValue+=1;
	}

	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

SingleEv_Handler_t PRV_GASPI_Event_Handlers[] =
{
	{GASPI_SIZE_EV,                 GASPI_Param},
	{GASPI_RANK_EV,                 GASPI_Param},
	{GASPI_NOTIFICATION_ID_EV,      GASPI_Param},
	{GASPI_QUEUE_ID_EV,             GASPI_Param},
	{GASPI_INIT_EV,                 GASPI_Event},
	{GASPI_TERM_EV,                 GASPI_Event},
	{GASPI_CONNECT_EV,              GASPI_Event},
	{GASPI_DISCONNECT_EV,           GASPI_Event},
	{GASPI_GROUP_CREATE_EV,         GASPI_Event},
	{GASPI_GROUP_ADD_EV,            GASPI_Event},
	{GASPI_GROUP_COMMIT_EV,         GASPI_Event},
	{GASPI_GROUP_DELETE_EV,         GASPI_Event},
	{GASPI_SEGMENT_ALLOC_EV,        GASPI_Event},
	{GASPI_SEGMENT_REGISTER_EV,     GASPI_Event},
	{GASPI_SEGMENT_CREATE_EV,       GASPI_Event},
	{GASPI_SEGMENT_BIND_EV,         GASPI_Event},
	{GASPI_SEGMENT_USE_EV,          GASPI_Event},
	{GASPI_SEGMENT_DELETE_EV,       GASPI_Event},
	{GASPI_WRITE_EV,                GASPI_Event},
	{GASPI_READ_EV,                 GASPI_Event},
	{GASPI_WAIT_EV,                 GASPI_Event},
	{GASPI_NOTIFY_EV,               GASPI_Event},
	{GASPI_NOTIFY_WAITSOME_EV,      GASPI_Event},
	{GASPI_NOTIFY_RESET_EV,         GASPI_Event},
	{GASPI_WRITE_NOTIFY_EV,         GASPI_Event},
	{GASPI_WRITE_LIST_EV,           GASPI_Event},
	{GASPI_WRITE_LIST_NOTIFY_EV,    GASPI_Event},
	{GASPI_READ_LIST_EV,            GASPI_Event},
	{GASPI_PASSIVE_SEND_EV,         GASPI_Event},
	{GASPI_PASSIVE_RECEIVE_EV,      GASPI_Event},
	{GASPI_ATOMIC_FETCH_ADD_EV,     GASPI_Event},
	{GASPI_ATOMIC_COMPARE_SWAP_EV,  GASPI_Event},
	{GASPI_BARRIER_EV,              GASPI_Event},
	{GASPI_ALLREDUCE_EV,            GASPI_Event},
	{GASPI_ALLREDUCE_USER_EV,       GASPI_Event},
	{NULL_EV,                          NULL}
};
