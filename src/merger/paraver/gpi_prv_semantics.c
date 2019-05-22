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

#include "gpi_prv_events.h"

static int
GPI_Event(event_t *current_event, unsigned long long current_time, unsigned cpu,
    unsigned ptask, unsigned task, unsigned thread, FileSet_t *fset)
{
	unsigned int EvType, nEvType;
	unsigned long long EvValue, nEvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent(current_event);
	EvValue = Get_EvValue(current_event);

	switch (EvType)
	{
		case GPI_INIT_EV:
		case GPI_TERM_EV:
			Switch_State(STATE_INITFINI, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GPI_BARRIER_EV:
			Switch_State(STATE_BARRIER, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GPI_WRITE_EV:
		case GPI_READ_EV:
		case GPI_WAIT_EV:
		case GPI_NOTIFY_EV:
		case GPI_NOTIFY_WAITSOME_EV:
		case GPI_NOTIFY_RESET_EV:
		case GPI_WRITE_NOTIFY_EV:
		case GPI_WRITE_LIST_EV:
		case GPI_WRITE_LIST_NOTIFY_EV:
		case GPI_READ_LIST_EV:
			Switch_State(STATE_1SIDED, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GPI_CONNECT_EV:
		case GPI_DISCONNECT_EV:
		case GPI_ALLREDUCE_EV:
			Switch_State(STATE_SYNC, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GPI_GROUP_CREATE_EV:
		case GPI_GROUP_ADD_EV:
		case GPI_GROUP_COMMIT_EV:
		case GPI_GROUP_DELETE_EV:
			Switch_State(STATE_MIXED, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
		case GPI_SEGMENT_ALLOC_EV:
		case GPI_SEGMENT_REGISTER_EV:
		case GPI_SEGMENT_CREATE_EV:
		case GPI_SEGMENT_BIND_EV:
		case GPI_SEGMENT_USE_EV:
		case GPI_SEGMENT_DELETE_EV:
			Switch_State(STATE_ALLOCMEM, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_state(cpu, ptask, task, thread, current_time);
			break;
	}

	Translate_GPI_Operation(EvType, EvValue, &nEvType, &nEvValue);
	trace_paraver_event(cpu, ptask, task, thread, current_time, nEvType, nEvValue);

	return 0;
}

static int
GPI_Param(event_t *current_event, unsigned long long current_time, unsigned cpu,
    unsigned ptask, unsigned task, unsigned thread, FileSet_t *fset)
{
	unsigned int EvType, nEvType;
	unsigned long long EvValue, nEvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent(current_event);
	EvValue = Get_EvValue(current_event);

	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

SingleEv_Handler_t PRV_GPI_Event_Handlers[] =
{
	{GPI_SIZE_EV,              GPI_Param},
	{GPI_RANK_EV,              GPI_Param},
	{GPI_INIT_EV,              GPI_Event},
	{GPI_TERM_EV,              GPI_Event},
	{GPI_CONNECT_EV,           GPI_Event},
	{GPI_DISCONNECT_EV,        GPI_Event},
	{GPI_GROUP_CREATE_EV,      GPI_Event},
	{GPI_GROUP_ADD_EV,         GPI_Event},
	{GPI_GROUP_COMMIT_EV,      GPI_Event},
	{GPI_GROUP_DELETE_EV,      GPI_Event},
	{GPI_SEGMENT_ALLOC_EV,     GPI_Event},
	{GPI_SEGMENT_REGISTER_EV,  GPI_Event},
	{GPI_SEGMENT_CREATE_EV,    GPI_Event},
	{GPI_SEGMENT_BIND_EV,      GPI_Event},
	{GPI_SEGMENT_USE_EV,       GPI_Event},
	{GPI_SEGMENT_DELETE_EV,    GPI_Event},
	{GPI_WRITE_EV,             GPI_Event},
	{GPI_READ_EV,              GPI_Event},
	{GPI_WAIT_EV,              GPI_Event},
	{GPI_NOTIFY_EV,            GPI_Event},
	{GPI_NOTIFY_WAITSOME_EV,   GPI_Event},
	{GPI_NOTIFY_RESET_EV,      GPI_Event},
	{GPI_WRITE_NOTIFY_EV,      GPI_Event},
	{GPI_WRITE_LIST_EV,        GPI_Event},
	{GPI_WRITE_LIST_NOTIFY_EV, GPI_Event},
	{GPI_READ_LIST_EV,         GPI_Event},
	{GPI_BARRIER_EV,           GPI_Event},
	{GPI_ALLREDUCE_EV,         GPI_Event},
	{NULL_EV,                       NULL}
};
