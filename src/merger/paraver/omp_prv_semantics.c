/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
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
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State ((EvValue==JOIN_WAIT_VAL)?STATE_BARRIER:STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

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
	unsigned int EvType;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_RUNNING, (EvValue != EVT_END), ptask, task, thread);

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
	{ NULL_EV, NULL }
};

