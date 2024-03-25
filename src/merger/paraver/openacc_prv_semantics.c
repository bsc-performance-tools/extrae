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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "file_set.h"
#include "paraver_generator.h"
#include "paraver_state.h"
#include "openacc_prv_semantics.h"

static int Get_State (unsigned int EvValue)
{
	switch (EvValue)
	{
		case OPENACC_INIT_VAL:
			return STATE_INITFINI;
		case OPENACC_ENTER_DATA_VAL:
		case OPENACC_EXIT_DATA_VAL:
			return STATE_ALLOCMEM;
		case OPENACC_UPDATE_VAL:
			return STATE_MEMORY_XFER;
		case OPENACC_COMPUTE_VAL:
			return STATE_CONFACCEL;
		case OPENACC_WAIT_VAL:
			return STATE_SYNC;
		default:
			return STATE_OTHERS;
	}
}

static int
OpenACC_Event(event_t *current_event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);
	unsigned int  EvType  = Get_EvEvent(current_event);
	unsigned long EvValue = Get_EvValue(current_event);
	unsigned long EvParam = Get_EvParam(current_event);

	Switch_State (Get_State(EvParam), (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state(cpu, ptask, task, thread, current_time);
	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, ((EvValue == EVT_BEGIN) ? EvParam : EVT_END));

	return 0;
}

static int
OpenACC_Data_Event(event_t *current_event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);
	unsigned int  EvType  = Get_EvEvent(current_event);
	unsigned long EvValue = Get_EvValue(current_event);
	unsigned long EvParam = Get_EvParam(current_event);

	switch(EvParam)
	{
		case OPENACC_ENQUEUE_UPLOAD_VAL:
		case OPENACC_ENQUEUE_DOWNLOAD_VAL:
			Switch_State(STATE_MEMORY_XFER, (EvValue != EVT_END), ptask, task, thread);
			trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, ((EvValue == EVT_BEGIN) ? EvParam : EVT_END));
			break;
		default:
			trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, EvValue);
	}

	return 0;
}

static int
OpenACC_Launch_Event(event_t *current_event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);
	unsigned int  EvType  = Get_EvEvent(current_event);
	unsigned long EvValue = Get_EvValue(current_event);
	unsigned long EvParam = Get_EvParam(current_event);

	Switch_State(STATE_CONFACCEL, (EvValue != EVT_END), ptask, task, thread);
	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, ((EvValue == EVT_BEGIN) ? EvParam : EVT_END));

	return 0;
}

SingleEv_Handler_t PRV_OPENACC_Event_Handlers[] = {
	{OPENACC_EV, OpenACC_Event},
	{OPENACC_DATA_EV, OpenACC_Data_Event},
	{OPENACC_LAUNCH_EV, OpenACC_Launch_Event},
	{NULL_EV, NULL }
};
