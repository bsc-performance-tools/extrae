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
	trace_paraver_event (cpu, ptask, task, thread, current_time, OMPFUNC_EV, EvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time, OMPFUNC_LINE_EV, EvValue);

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

	/* Add the instantiated task to the list of known addresses, and emit its
	   reference for matching in final tracefile */

#if defined(HAVE_BFD)
	if (get_option_merge_SortAddresses())
	{
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_LINE);
	}
#endif

	trace_paraver_event (cpu, ptask, task, thread, current_time, TASKFUNC_INST_EV, EvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time, TASKFUNC_INST_LINE_EV, EvValue);

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

	Switch_State (STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

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
	{ NULL_EV, NULL }
};

