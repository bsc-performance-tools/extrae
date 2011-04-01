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

#include <config.h>

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "file_set.h"
#include "object_tree.h"
#include "misc_prv_semantics.h"
#include "trace_to_prv.h"
#include "misc_prv_events.h"
#include "semantics.h"
#include "paraver_generator.h"
#include "communication_queues.h"
#include "trace_communication.h"
#include "addresses.h"
#include "options.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#ifdef HAVE_BFD
# include "addr2info.h" 
#endif

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
# include "timesync.h"
#endif

#include "events.h"
#include "paraver_state.h"

int MPI_Caller_Multiple_Levels_Traced = FALSE;
int *MPI_Caller_Labels_Used = NULL;

int Sample_Caller_Multiple_Levels_Traced = FALSE;
int *Sample_Caller_Labels_Used = NULL;

int Rusage_Events_Found = FALSE;
int GetRusage_Labels_Used[RUSAGE_EVENTS_COUNT];
int Memusage_Events_Found = FALSE;
int Memusage_Labels_Used[MEMUSAGE_EVENTS_COUNT];
int MPI_Stats_Events_Found = FALSE;
int MPI_Stats_Labels_Used[MPI_STATS_EVENTS_COUNT];
int PACX_Stats_Events_Found = FALSE;
int PACX_Stats_Labels_Used[MPI_STATS_EVENTS_COUNT];

int MaxClusterId = 0; /* Marks the maximum cluster id assigned in the mpits */

/******************************************************************************
 ***  Flush_Event
 ******************************************************************************/

static int Flush_Event (event_t * current_event,
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

	Switch_State (STATE_FLUSH, (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

/******************************************************************************
 ***  Read_Event
 ******************************************************************************/

static int ReadWrite_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	unsigned long EvParam;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);
	EvParam = Get_EvParam (current_event);

	Switch_State (STATE_IO, (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	switch (EvValue)
	{
		case EVT_BEGIN:
		trace_paraver_event (cpu, ptask, task, thread, current_time, IOSIZE_EV, EvParam);
		break;
		case EVT_END:
		break;
	}
	return 0;
}

/******************************************************************************
 ***   Tracing_Event
 ******************************************************************************/

static int Tracing_Event (event_t * current_event,
                          unsigned long long current_time,
                          unsigned int cpu,
                          unsigned int ptask,
                          unsigned int task,
                          unsigned int thread,
                          FileSet_t *fset)
{
	unsigned int EvType, EvValue, i;
	struct task_t * task_info;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	task_info = GET_TASK_INFO(ptask, task);
	task_info -> tracing_disabled = TRUE;

	/* Mark all threads of the current task as not tracing */
	for (i = 0; i < task_info->nthreads; i++)
	{
		Switch_State (STATE_NOT_TRACING, (EvValue == EVT_END), ptask, task, i+1);

		trace_paraver_state (cpu, ptask, task, i + 1, current_time);
	}

	/* Only the task writes the event */
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

/******************************************************************************
 ***  Appl_Event
 ******************************************************************************/

static int Appl_Event (event_t * current_event,
                       unsigned long long current_time,
                       unsigned int cpu,
                       unsigned int ptask,
                       unsigned int task,
                       unsigned int thread,
                       FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	if (EvValue == EVT_END)
		Pop_State (STATE_ANY, ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

/******************************************************************************
 ***  User_Event
 ******************************************************************************/

static int User_Event (event_t * current_event,
                       unsigned long long current_time,
                       unsigned int cpu,
                       unsigned int ptask,
                       unsigned int task,
                       unsigned int thread,
                       FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvValue (current_event);     /* Value is the user event type.  */
	EvValue = Get_EvMiscParam (current_event); /* Param is the user event value. */

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

/******************************************************************************
 ***  MPI_Caller_Event
 ******************************************************************************/

static int MPI_Caller_Event (event_t * current_event,
                             unsigned long long current_time,
                             unsigned int cpu,
                             unsigned int ptask,
                             unsigned int task,
                             unsigned int thread,
                             FileSet_t *fset)
{
	int i, deepness;	
	UINT64 EvValue = Get_EvValue(current_event);
	UNREFERENCED_PARAMETER(fset);

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	deepness = Get_EvEvent(current_event) - CALLER_EV;
	if (deepness > 0) 
	{
		MPI_Caller_Multiple_Levels_Traced = TRUE;	
		if (MPI_Caller_Labels_Used == NULL) 
		{
			MPI_Caller_Labels_Used = (int *)malloc(sizeof(int)*MAX_CALLERS);
			for (i = 0; i < MAX_CALLERS; i++) 
			{
				MPI_Caller_Labels_Used[i] = FALSE;
			}
		}
		if (MPI_Caller_Labels_Used != NULL) 
		{
			MPI_Caller_Labels_Used [deepness-1] = TRUE; 
		}
	}

#if defined(HAVE_BFD)
	if (get_option_merge_SortAddresses())
	{
		AddressCollector_Add (&CollectedAddresses, EvValue, ADDR2MPI_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, EvValue, ADDR2MPI_LINE);
	}
#endif

	trace_paraver_event (cpu, ptask, task, thread, current_time, CALLER_EV+deepness, EvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time, CALLER_LINE_EV+deepness, EvValue);

	return 0;
}

/******************************************************************************
 ***  GetRusage_Event
 ******************************************************************************/

static int GetRusage_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	int i;
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvValue (current_event);       /* Value is the user event type.  */
	EvValue = Get_EvMiscParam (current_event);   /* Param is the user event value. */

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, RUSAGE_BASE+EvType, EvValue);

	if (!Rusage_Events_Found) 
	{
		Rusage_Events_Found = TRUE;
		for (i=0; i<RUSAGE_EVENTS_COUNT; i++)
		{
			GetRusage_Labels_Used[i] = FALSE;
		}
	}
	GetRusage_Labels_Used[EvType] = TRUE;

	return 0;
}

/******************************************************************************
 ***  Memusage_Event
 ******************************************************************************/

static int Memusage_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
    int i;
    unsigned int EvType, EvValue;
    UNREFERENCED_PARAMETER(fset);

    EvType  = Get_EvValue (current_event);       /* Value is the user event type.  */
    EvValue = Get_EvMiscParam (current_event);   /* Param is the user event value. */

    trace_paraver_state (cpu, ptask, task, thread, current_time);
    trace_paraver_event (cpu, ptask, task, thread, current_time, MEMUSAGE_BASE+EvType, EvValue);

    if (!Memusage_Events_Found)
    {
        Memusage_Events_Found = TRUE;
        for (i=0; i<MEMUSAGE_EVENTS_COUNT; i++)
        {
            Memusage_Labels_Used[i] = FALSE;
        }
    }
    Memusage_Labels_Used[EvType] = TRUE;

    return 0;
}

/******************************************************************************
 ***  MPI_Stats_Event
 ******************************************************************************/
static int MPI_Stats_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	int i;
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvValue (current_event);     /* Value is the event type.  */
	EvValue = Get_EvMiscParam (current_event); /* Param is the event value. */

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, MPI_STATS_BASE+EvType, EvValue);

	if (!MPI_Stats_Events_Found)
	{
		MPI_Stats_Events_Found = TRUE;
		for (i=0; i<MPI_STATS_EVENTS_COUNT; i++)
		{
			MPI_Stats_Labels_Used[i] = FALSE;
		}
	}
	MPI_Stats_Labels_Used[EvType] = TRUE;

	return 0;
}


/******************************************************************************
 ***  PACX_Stats_Event
 ******************************************************************************/
static int PACX_Stats_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	int i;
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvValue (current_event);     /* Value is the event type.  */
	EvValue = Get_EvMiscParam (current_event); /* Param is the event value. */

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, PACX_STATS_BASE+EvType, EvValue);

	if (!PACX_Stats_Events_Found)
	{
		PACX_Stats_Events_Found = TRUE;
		for (i=0; i<PACX_STATS_EVENTS_COUNT; i++)
		{
			PACX_Stats_Labels_Used[i] = FALSE;
		}
	}
	PACX_Stats_Labels_Used[EvType] = TRUE;

	return 0;
}


/******************************************************************************
 ***  USRFunction_Event
 ******************************************************************************/
static int USRFunction_Event (event_t * current,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t *fset )
{
	unsigned int EvType;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current);
	EvValue = Get_EvValue (current);

	Switch_State (STATE_RUNNING, (EvValue != EVT_END), ptask, task, thread);

#if defined(HAVE_BFD)
	if (get_option_merge_SortAddresses() && EvValue != 0)
	{
		AddressCollector_Add (&CollectedAddresses, EvValue, ADDR2UF_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, EvValue, ADDR2UF_LINE);
	}
#endif

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, USRFUNC_EV, EvValue);
	if (EvValue != 0)
		trace_paraver_event (cpu, ptask, task, thread, current_time, USRFUNC_LINE_EV, EvValue);

	return 0;
}

/******************************************************************************
 ***  Sampling_Caller_Event
 ******************************************************************************/
static int Sampling_Caller_Event (event_t * current,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned EvType;
	unsigned LINE_EV_DELTA;
	unsigned int EvTypeDelta, i;
	UINT64  EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType = Get_EvEvent(current);
	EvValue = Get_EvValue (current);

	EvTypeDelta = EvType - SAMPLING_EV;
	LINE_EV_DELTA = SAMPLING_LINE_EV - SAMPLING_EV;

	if (Sample_Caller_Labels_Used == NULL) 
	{
		Sample_Caller_Labels_Used = (int *)malloc(sizeof(int)*MAX_CALLERS);
		for (i = 0; i < MAX_CALLERS; i++) 
			Sample_Caller_Labels_Used[i] = FALSE;
	}	     
	if (Sample_Caller_Labels_Used != NULL) 
		Sample_Caller_Labels_Used [EvTypeDelta] = TRUE; 
	  
	if (EvValue != 0)
	{

#if defined(HAVE_BFD)
		if (get_option_merge_SortAddresses())
		{
			if (EvTypeDelta == 0)
			{
				/* If depth == 0 (in EvTypeDelta) addresses are taken from the overflow
				   routine which points to the "originating" address */
				AddressCollector_Add (&CollectedAddresses, EvValue, ADDR2SAMPLE_FUNCTION);
				AddressCollector_Add (&CollectedAddresses, EvValue, ADDR2SAMPLE_LINE);
			}
			else
			{
				/* If depth != 0 (in EvTypeDelta), addresses are taken from the callstack
				   and point to the next instruction, so substract 1 */
				AddressCollector_Add (&CollectedAddresses, EvValue-1, ADDR2SAMPLE_FUNCTION);
				AddressCollector_Add (&CollectedAddresses, EvValue-1, ADDR2SAMPLE_LINE);
			}
		}
#endif

		trace_paraver_state (cpu, ptask, task, thread, current_time);

		if (EvTypeDelta == 0)
		{
			/* If depth == 0 (in EvTypeDelta) addresses are taken from the overflow
			   routine which points to the "originating" address */
			trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
			trace_paraver_event (cpu, ptask, task, thread, current_time, EvType+LINE_EV_DELTA, EvValue);
		}
		else
		{
			/* If depth != 0 (in EvTypeDelta), addresses are taken from the callstack and
			   point to the next instruction, so substract 1 */
			trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue-1);
			trace_paraver_event (cpu, ptask, task, thread, current_time, EvType+LINE_EV_DELTA, EvValue-1);
		}
	}

	return 0;
}

#if USE_HARDWARE_COUNTERS
static int Set_Overflow_Event (event_t * current_event,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t *fset )
{
	UNREFERENCED_PARAMETER(fset);

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	HardwareCounters_SetOverflow (ptask, task, thread, current_event);

	return 0;
}
#endif

static int Tracing_Mode_Event (event_t * current_event,
    unsigned long long current_time, unsigned int cpu, unsigned int ptask,
    unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	Initialize_Trace_Mode_States (cpu, ptask, task, thread, EvValue);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

#if USE_HARDWARE_COUNTERS

static int Evt_CountersDefinition (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	unsigned nthreads;
	unsigned i;
	int newSet = Get_EvValue(current_event);
	long long *HWCIds = Get_EvHWCVal(current_event);
	UNREFERENCED_PARAMETER(fset);
	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(thread);
	UNREFERENCED_PARAMETER(cpu);

	/* The hardware counter set definition exists only on the master thread.
	   We replicate them to all the threads as they appear */
	nthreads = obj_table[ptask-1].tasks[task-1].nthreads;
	for (i = 1; i <= nthreads; i++)
		HardwareCounters_NewSetDefinition(ptask, task, i, newSet, HWCIds);

	return 0;
}

/******************************************************************************
 **      Function name : ResetCounters
 **      Description :
 ******************************************************************************/

static void ResetCounters (int ptask, int task)
{
	int thread, cnt;

	obj_table[ptask].tasks[task].tracing_disabled = FALSE;

	for (thread = 0; thread < obj_table[ptask].tasks[task].nthreads; thread++)
		for (cnt = 0; cnt < MAX_HWC; cnt++)
			obj_table[ptask].tasks[task].threads[thread].counters[cnt] = 0LL;
}

/******************************************************************************
 **      Function name : Evt_SetCounters
 **      Description :
 ******************************************************************************/

int HWC_Change_Ev (
   int newSet,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread)
{
	int i;
	unsigned int hwctype[MAX_HWC+1];
	unsigned long long hwcvalue[MAX_HWC+1];
	unsigned int prev_hwctype[MAX_HWC];
	struct thread_t * Sthread;
	Sthread = GET_THREAD_INFO(ptask, task, thread);

	int oldSet = HardwareCounters_GetCurrentSet(ptask, task, thread);
	int *oldIds = HardwareCounters_GetSetIds(ptask, task, thread, oldSet);

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	/* Store which were the counters being read before (they're overwritten with the new set at HardwareCounters_Change) */
	for (i=0; i<MAX_HWC; i++)
#if defined(PMAPI_COUNTERS)
		prev_hwctype[i] = HWC_COUNTER_TYPE(i, oldsIds[i]);
#else
		prev_hwctype[i] = HWC_COUNTER_TYPE(oldIds[i]);
#endif

	ResetCounters (ptask-1, task-1);
	HardwareCounters_Change (ptask, task, thread, newSet, hwctype, hwcvalue);

	for (i = 0; i < MAX_HWC+1; i++)
	{
		if (NO_COUNTER != hwctype[i])
		{
			int found = FALSE, k = 0;

			/* Check the current counter (hwctype[i]) did not appear on the previous set. We don't want
			 * it to appear twice in the same timestamp. This may happen because the HWC_CHANGE_EV is traced
			 * right after the last valid emission of counters with the previous set, at the same timestamp.
			 */

			while ((!found) && (k < MAX_HWC))
			{
				if (hwctype[i] == prev_hwctype[k]) found = TRUE;
				k ++;
			}

			if (!found)
				trace_paraver_event (cpu, ptask, task, thread, current_time, hwctype[i], hwcvalue[i]);
		}
	}
	return 0;
}

static int Evt_SetCounters (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
    UNREFERENCED_PARAMETER(fset);
    unsigned int newSet = Get_EvValue (current_event);

	return HWC_Change_Ev (newSet, current_time, cpu, ptask, task, thread);
}

#endif /* HARDWARE_COUNTERS */

static int CPU_Burst_Event (
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

	Switch_State (STATE_RUNNING, (EvValue == EVT_BEGIN), ptask, task, thread);
	trace_paraver_state (cpu, ptask, task, thread, current_time);

/* We don't trace this event in CPU Burst mode. This is just for debugging purposes
   trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue); 
*/

	return 0;
}


/******************************************************************************
 **      Function name : traceCounters
 **      Description :
 ******************************************************************************/

static int traceCounters (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	UNREFERENCED_PARAMETER(fset);
	UNREFERENCED_PARAMETER(current_event);

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	return 0;
}

static int SetTracing_Event (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	UNREFERENCED_PARAMETER(fset);

	if (!Get_EvValue (current_event))
	{
		Push_State (STATE_NOT_TRACING, ptask, task, thread);
		trace_paraver_state (cpu, ptask, task, thread, current_time);

		/* Mark when the tracing is disabled! */
		EnabledTasks_time[ptask - 1][task - 1] = current_time;
	}
/*
   else if (Top_State (ptask, task, thread) == STATE_NOT_TRACING)
   {
      Pop_State (ptask, task, thread);
   }
*/
	else 
	{
		Pop_State (STATE_NOT_TRACING, ptask, task, thread);
	}

	EnabledTasks[ptask - 1][task - 1] = Get_EvValue (current_event);

	return 0;
}

static int MRNet_Event (event_t * current_event,
    unsigned long long current_time, unsigned int cpu, unsigned int ptask,
    unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int Clustering_Event (event_t * current_event,
    unsigned long long current_time, unsigned int cpu, unsigned int ptask,
    unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	MaxClusterId = MAX(MaxClusterId, EvValue);

	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int Spectral_Event (event_t * current_event,
    unsigned long long current_time, unsigned int cpu, unsigned int ptask,
    unsigned int task, unsigned int thread, FileSet_t *fset)
{
    unsigned int EvType, EvValue;
    UNREFERENCED_PARAMETER(fset);

    EvType  = Get_EvEvent (current_event);
    EvValue = Get_EvValue (current_event);

    trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

    return 0;
}

/******************************************************************************
 ***  User_Send_Event
 ******************************************************************************/

static int User_Send_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned recv_thread;
	struct thread_t *thread_info;
	struct task_t *task_info, *task_info_partner;
	event_t * recv_begin, * recv_end;
	UNREFERENCED_PARAMETER(cpu);

	thread_info = GET_THREAD_INFO(ptask, task, 1);
	task_info = GET_TASK_INFO(ptask, task);


	if (MatchComms_Enabled(ptask, task, thread))
	{
		if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
		{
			task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(current_event)+1);

			CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread, Get_EvAux(current_event));

			if (recv_begin == NULL || recv_end == NULL)
			{
				off_t position;

				position = WriteFileBuffer_getPosition (obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb);
				CommunicationQueues_QueueSend (task_info->send_queue, current_event, current_event, position, thread, Get_EvAux(current_event));
				trace_paraver_unmatched_communication (1, ptask, task, thread, current_time, Get_EvTime(current_event), 1, ptask, Get_EvTarget(current_event)+1, recv_thread, Get_EvSize(current_event), Get_EvTag(current_event));
			}
			else
			{
				trace_communicationAt (ptask, task, thread, 1+Get_EvTarget(current_event), recv_thread, current_event, current_event, recv_begin, recv_end, FALSE, 0);
			}
		}
#if defined(PARALLEL_MERGE)
		else
			trace_pending_communication (ptask, task, thread, current_event, current_event, Get_EvTarget (current_event));
#endif
	}

	return 0;
}

/******************************************************************************
 ***  Recv_Event
 ******************************************************************************/

static int User_Recv_Event (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	event_t *send_begin, *send_end;
	off_t send_position;
	unsigned send_thread;
	struct thread_t *thread_info;
	struct task_t *task_info, *task_info_partner;
	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(current_time);

	thread_info = GET_THREAD_INFO(ptask, task, 1);
	task_info = GET_TASK_INFO(ptask, task);

	if (MatchComms_Enabled(ptask, task, thread))
	{
		if (isTaskInMyGroup (fset, Get_EvTarget(current_event)))
		{
			task_info_partner = GET_TASK_INFO(ptask, Get_EvTarget(current_event)+1);

			CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (current_event), &send_begin, &send_end, &send_position, &send_thread, Get_EvAux(current_event));

			if (NULL == send_begin || NULL == send_end)
			{
				CommunicationQueues_QueueRecv (task_info->recv_queue, current_event, current_event, thread, Get_EvAux(current_event));
			}
			else if (NULL != send_begin && NULL != send_end)
			{
				trace_communicationAt (ptask, 1+Get_EvTarget(current_event), send_thread, task, thread, send_begin, send_end, current_event, current_event, TRUE, send_position);
			}
			else
				fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
		}
#if defined(PARALLEL_MERGE)
		else
		{
			UINT64 log_r, phy_r;

			log_r = TIMESYNC (task-1, Get_EvTime(current_event));
			phy_r = TIMESYNC (task-1, Get_EvTime(current_event));
			AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), task-1,
			  Get_EvTarget(current_event), fset);
		}
#endif
	}

	return 0;
}


SingleEv_Handler_t PRV_MISC_Event_Handlers[] = {
	{ FLUSH_EV, Flush_Event },
	{ READ_EV, ReadWrite_Event },
	{ WRITE_EV, ReadWrite_Event },
	{ APPL_EV, Appl_Event },
	{ USER_EV, User_Event },
	{ HWC_EV, traceCounters },
#if USE_HARDWARE_COUNTERS
	{ HWC_DEF_EV, Evt_CountersDefinition },
	{ HWC_CHANGE_EV, Evt_SetCounters },
	{ HWC_SET_OVERFLOW_EV, Set_Overflow_Event },
#else
	{ HWC_DEF_EV, SkipHandler },
	{ HWC_CHANGE_EV, SkipHandler },
	{ HWC_SET_OVERFLOW_EV, SkipHandler },
#endif
	{ TRACING_EV, Tracing_Event },
	{ SET_TRACE_EV, SetTracing_Event },
	{ CPU_BURST_EV, CPU_Burst_Event },
	{ RUSAGE_EV, GetRusage_Event },
	{ MEMUSAGE_EV, Memusage_Event },
	{ MPI_STATS_EV, MPI_Stats_Event },
	{ PACX_STATS_EV, PACX_Stats_Event },
	{ USRFUNC_EV, USRFunction_Event },
	{ TRACING_MODE_EV, Tracing_Mode_Event },
	{ MRNET_EV, MRNet_Event },
	{ CLUSTER_ID_EV, Clustering_Event },
	{ SPECTRAL_PERIOD_EV, Spectral_Event },
	{ USER_SEND_EV, User_Send_Event},
	{ USER_RECV_EV, User_Recv_Event},
	{ NULL_EV, NULL }
};

RangeEv_Handler_t PRV_MISC_Range_Handlers[] = {
	{ CALLER_EV, CALLER_EV + MAX_CALLERS, MPI_Caller_Event },
	{ SAMPLING_EV, SAMPLING_EV + MAX_CALLERS, Sampling_Caller_Event },
	{ NULL_EV, NULL_EV, NULL }
};

