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
#include "object_tree.h"
#include "omp_prv_semantics.h"
#include "trace_to_prv.h"
#include "omp_prv_events.h"
#include "semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "addresses.h"
#include "options.h"

#if USE_HARDWARE_COUNTERS
#include "HardwareCounters.h"
#endif

#ifdef HAVE_BFD
# include "addr2info.h" 
#endif

#include "record.h"
#include "events.h"

/******************************************************************************
 ***  WorkSharing_Event
 ******************************************************************************/

static int WorkSharing_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int Parallel_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int Join_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue, EvParam;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);
	EvParam = Get_EvParam (current_event);

	Switch_State ((EvParam==JOIN_WAIT_VAL)?STATE_BARRIER:STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int Work_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int OpenMP_Function_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_RUNNING, (EvValue != EVT_END), ptask, task, thread);

#if defined(HAVE_BFD)
	if (get_option_merge_SortAddresses())
	{
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_LINE);
	}
#endif

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	if (Get_EvEvent(current_event) == OMPFUNC_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time, OMPFUNC_EV, EvValue);
		trace_paraver_event (cpu, ptask, task, thread, current_time, OMPFUNC_LINE_EV, EvValue);
	}
	else if (Get_EvEvent(current_event) == TASKFUNC_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time, TASKFUNC_EV, EvValue);
		trace_paraver_event (cpu, ptask, task, thread, current_time, TASKFUNC_LINE_EV, EvValue);
	}

	return 0;
}

static int BarrierOMP_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_BARRIER, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int Critical_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_SYNC, ((EvValue == LOCK_VAL) || (EvValue == UNLOCK_VAL)), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
	if (EvType == NAMEDCRIT_EV && (EvValue == LOCKED_VAL || EvValue == UNLOCKED_VAL))
	{
		/* At the entry of lock and unlock of a named critical, emit also the address */
		trace_paraver_event (cpu, ptask, task, thread, current_time, NAMEDCRIT_NAME_EV, Get_EvParam(current_event));
	}

	return 0;
}

static int SetGetNumThreads_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int Task_Event (
   event_t * event,
   unsigned long long time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	UNREFERENCED_PARAMETER(fset);

#if defined(HAVE_BFD)
	/* Add the instantiated task to the list of known addresses, and emit its
	   reference for matching in final tracefile */

	if (get_option_merge_SortAddresses())
	{
		AddressCollector_Add (&CollectedAddresses, ptask, task,
		  Get_EvValue(event),ADDR2OMP_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, ptask, task,
		  Get_EvValue(event), ADDR2OMP_LINE);
	}
#endif

	Switch_State (STATE_OVHD, Get_EvValue(event) != EVT_END, ptask, task, thread);
	trace_paraver_state (cpu, ptask, task, thread, time);
	trace_paraver_event (cpu, ptask, task, thread, time, TASKFUNC_INST_EV,
		Get_EvValue(event));
	trace_paraver_event (cpu, ptask, task, thread, time, TASKFUNC_INST_LINE_EV,
		Get_EvValue(event));

	return 0;
}

static int Taskwait_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_SYNC, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int TaskGroup_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	if (EvType == TASKGROUP_START_EV)
		Switch_State (STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);
	else if (EvType == TASKGROUP_END_EV)
		Switch_State (STATE_SYNC, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	if (EvType == TASKGROUP_START_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  TASKGROUP_START_EV, EvValue?1:0);
		if (EvValue)
			trace_paraver_event (cpu, ptask, task, thread, current_time,
			  TASKGROUP_INGROUP_DEEP_EV, EVT_BEGIN);
	}
	else if (EvType == TASKGROUP_END_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  TASKGROUP_START_EV, EvValue?2:0);
		if (!EvValue)
			trace_paraver_event (cpu, ptask, task, thread, current_time,
			  TASKGROUP_INGROUP_DEEP_EV, EVT_END);
	}


	return 0;
}

static int OMPT_event (event_t * current_event,
	unsigned long long current_time,
	unsigned cpu,
	unsigned ptask,
	unsigned task,
	unsigned thread,
	FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	switch (EvType)
	{
		case OMPT_CRITICAL_EV:
		case OMPT_ATOMIC_EV:
		case OMPT_SINGLE_EV:
		case OMPT_MASTER_EV:
		Switch_State (STATE_BARRIER, (EvValue != EVT_END), ptask, task, thread);
		trace_paraver_state (cpu, ptask, task, thread, current_time);
		break;
	}

	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int TaskID_Event (event_t * event,
	unsigned long long current_time,
	unsigned cpu,
	unsigned ptask,
	unsigned task,
	unsigned thread,
	FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);
	trace_paraver_event (cpu, ptask, task, thread, current_time, TASKID_EV,
	  Get_EvParam(event));
	return 0;
}

static int OMPT_TaskGroup_Event (event_t *event,
	unsigned long long time,
	unsigned cpu,
	unsigned ptask,
	unsigned task,
	unsigned thread,
	FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);

	trace_paraver_event (cpu, ptask, task, thread, time,
	  TASKGROUP_INGROUP_DEEP_EV, Get_EvValue (event));

	return 0;
}

static int OMPT_dependence_Event (event_t *event,
	unsigned long long time,
	unsigned cpu,
	unsigned ptask,
	unsigned task,
	unsigned thread,
	FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);
	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(time);
	UNREFERENCED_PARAMETER(thread);

	task_t *task_info = GET_TASK_INFO(ptask, task);

	ThreadDependency_add (task_info->thread_dependencies, event);

	return 0;
}

struct TaskFunction_Event_Info_SetPredecessor
{
	unsigned long long time;
	unsigned cpu, ptask, task, thread;
};

struct TaskFunction_Event_Info_EmitDependencies
{
	unsigned long long time;
	unsigned cpu, ptask, task, thread;
	event_t *event;
};

static int TaskEvent_IfSetPredecessor (const void *dependency_event, void *userdata,
	void **predecessordata)
{
	const event_t *depevent = (const event_t*) dependency_event;
	struct TaskFunction_Event_Info_EmitDependencies *tfei =
		(struct TaskFunction_Event_Info_EmitDependencies*) userdata;
	event_t *checkevent = tfei->event;

	if (Get_EvParam(checkevent) == Get_EvParam(depevent))
	{
		struct TaskFunction_Event_Info_SetPredecessor *tfeisp =
		  (struct TaskFunction_Event_Info_SetPredecessor *)
		    malloc (sizeof(struct TaskFunction_Event_Info_SetPredecessor));
		if (tfeisp != NULL)
		{
			tfeisp->ptask = tfei->ptask;
			tfeisp->task = tfei->task;
			tfeisp->cpu = tfei->cpu;
			tfeisp->thread = tfei->thread;
			tfeisp->time = tfei->time;
			*predecessordata = tfeisp;
		}

		return TRUE;
	}
	else
		return FALSE;
}

static int TaskEvent_IfEmitDependencies (const void *dependency_event, 
	const void *predecessor_data, const void *userdata)
{
	struct TaskFunction_Event_Info_EmitDependencies *tfei =
		(struct TaskFunction_Event_Info_EmitDependencies*) userdata;
	const event_t *depevent = (const event_t*) dependency_event;
	const struct TaskFunction_Event_Info_SetPredecessor *preddata =
	  (const struct TaskFunction_Event_Info_SetPredecessor *) predecessor_data;
	event_t *checkevent = tfei->event;

	if (Get_EvNParam(depevent, 1) == Get_EvNParam(checkevent, 0))
	{
		trace_paraver_communication (
		  preddata->cpu, preddata->ptask, preddata->task,
		  preddata->thread, preddata->thread, preddata->time, preddata->time,
		  tfei->cpu, tfei->ptask, tfei->task, tfei->thread, tfei->thread,
		  tfei->time, tfei->time,
		  0, Get_EvValue(depevent),
		  0, 0);
	}

	return 0;
}

static int OMPT_TaskFunction_Event (
   event_t * event,
   unsigned long long time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	UNREFERENCED_PARAMETER(fset);

#if defined(HAVE_BFD)
	/* Add the instantiated task to the list of known addresses, and emit its
	   reference for matching in final tracefile */

	if (get_option_merge_SortAddresses())
	{
		AddressCollector_Add (&CollectedAddresses, ptask, task,
		  Get_EvParam(event),ADDR2OMP_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, ptask, task,
		  Get_EvParam(event), ADDR2OMP_LINE);
	}
#endif

	Switch_State (STATE_RUNNING, Get_EvValue(event) != EVT_END, ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, time);
	trace_paraver_event (cpu, ptask, task, thread, time, TASKFUNC_EV,
		Get_EvValue(event));
	trace_paraver_event (cpu, ptask, task, thread, time, TASKFUNC_LINE_EV,
		Get_EvValue(event));

	if (Get_EvValue(event) == EVT_END)
	{
		task_t * task_info = GET_TASK_INFO(ptask, task);
		struct TaskFunction_Event_Info_EmitDependencies data;
		data.time = time;
		data.cpu = cpu;
		data.ptask = ptask;
		data.task = task;
		data.thread = thread;
		data.event = event;

		ThreadDependency_processAll_ifMatchSetPredecessor (
		  task_info->thread_dependencies,
		  TaskEvent_IfSetPredecessor,
		  &data);
	}
	else
	{
		task_t * task_info = GET_TASK_INFO(ptask, task);
		struct TaskFunction_Event_Info_EmitDependencies data;
		data.time = time;
		data.cpu = cpu;
		data.ptask = ptask;
		data.task = task;
		data.thread = thread;
		data.event = event;

		ThreadDependency_processAll_ifMatchDelete (
		  task_info->thread_dependencies,
		  TaskEvent_IfEmitDependencies,
		  &data);
	}

	return 0;
}

static int OMP_Stats_Event (
   event_t * event,
   unsigned long long time,
   unsigned int cpu,
   unsigned int ptask, 
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	UNREFERENCED_PARAMETER(fset);

	trace_paraver_event (cpu, ptask, task, thread, time,
		OMP_STATS_BASE+Get_EvValue(event), Get_EvParam(event));

	return 0;
}

SingleEv_Handler_t PRV_OMP_Event_Handlers[] = {
	{ WSH_EV, WorkSharing_Event },
	{ PAR_EV, Parallel_Event },
	{ OMPFUNC_EV, OpenMP_Function_Event },
	{ BARRIEROMP_EV, BarrierOMP_Event },
	{ UNNAMEDCRIT_EV, Critical_Event },
	{ NAMEDCRIT_EV, Critical_Event },
	{ WORK_EV, Work_Event},
	{ JOIN_EV, Join_Event},
	{ OMPSETNUMTHREADS_EV, SetGetNumThreads_Event },
	{ OMPGETNUMTHREADS_EV, SetGetNumThreads_Event },
	{ TASK_EV, Task_Event },
	{ TASKWAIT_EV, Taskwait_Event },
	{ TASKFUNC_EV, OpenMP_Function_Event },
	{ OMPT_CRITICAL_EV, OMPT_event },
	{ OMPT_ATOMIC_EV, OMPT_event },
	{ OMPT_LOOP_EV, OMPT_event },
	{ OMPT_WORKSHARE_EV, OMPT_event },
	{ OMPT_SECTIONS_EV, OMPT_event },
	{ OMPT_SINGLE_EV, OMPT_event },
	{ OMPT_MASTER_EV, OMPT_event },
	{ TASKGROUP_START_EV, TaskGroup_Event },
	{ TASKGROUP_END_EV, TaskGroup_Event },
	{ TASKID_EV, TaskID_Event },
	{ OMPT_TASKGROUP_IN_EV, OMPT_TaskGroup_Event },
	{ OMPT_DEPENDENCE_EV, OMPT_dependence_Event },
	{ OMPT_TASKFUNC_EV, OMPT_TaskFunction_Event },
	{ OMP_STATS_EV, OMP_Stats_Event },
	{ NULL_EV, NULL }
};

