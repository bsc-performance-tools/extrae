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
#include "extrae_types.h"
#include "online_events.h"
#include "trace_mode.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#include "addr2info.h" 

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
int Syscall_Events_Found = FALSE;
int Syscall_Labels_Used[SYSCALL_EVENTS_COUNT];

unsigned int MaxClusterId = 0; /* Marks the maximum cluster id assigned in the mpits */

unsigned int MaxRepresentativePeriod = 0;
unsigned int HaveSpectralEvents = FALSE;

static int Get_State (unsigned int EvType)
{
	int state = 0;

	switch (EvType)
	{
		case MALLOC_EV:
		case MEMKIND_MALLOC_EV:
		case MEMKIND_POSIX_MEMALIGN_EV:
		case POSIX_MEMALIGN_EV:
		case REALLOC_EV:
		case MEMKIND_REALLOC_EV:
		case CALLOC_EV:
		case MEMKIND_CALLOC_EV:
		case KMPC_MALLOC_EV:
		case KMPC_CALLOC_EV:
		case KMPC_REALLOC_EV:
		case KMPC_ALIGNED_MALLOC_EV:
			state = STATE_ALLOCMEM;
	  break;
		case FREE_EV:
		case MEMKIND_FREE_EV:
		case KMPC_FREE_EV:
			state = STATE_FREEMEM;
		break;
		default:
			fprintf (stderr, "mpi2prv: Error! Unknown MPI event %d parsed at %s (%s:%d)\n",
			  EvType, __func__, __FILE__, __LINE__);
			fflush (stderr);
			exit (-1);
		break;
	}
	return state;
}

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
 ***  ReadWrite_Event
 ******************************************************************************/

static int ReadWrite_Event (event_t * event, unsigned long long time,
	unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
	FileSet_t *fset)
{
	unsigned int EvType;
	unsigned long EvValue, EvParam;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (event);
	EvValue = Get_EvValue (event);
	EvParam = Get_EvParam (event);

	if (EvValue == EVT_BEGIN || EvValue == EVT_END)
	{
		Switch_State (STATE_IO, (EvValue == EVT_BEGIN), ptask, task, thread);
		trace_paraver_state (cpu, ptask, task, thread, time);
	}

	if (EvValue != EVT_END)
	{
		int io_type;

		switch (Get_EvValue(event))
		{
			case EVT_BEGIN:
				switch(EvType)
				{
					case OPEN_EV:
						io_type = OPEN_VAL_EV;
						break;
					case FOPEN_EV:
						io_type = FOPEN_VAL_EV;
						break;
					case READ_EV:
						io_type = READ_VAL_EV;
						break;
					case WRITE_EV:
						io_type = WRITE_VAL_EV;
						break;
					case FREAD_EV:
						io_type = FREAD_VAL_EV;
						break;
					case FWRITE_EV:
						io_type = FWRITE_VAL_EV;
						break;
					case PREAD_EV:
						io_type = PREAD_VAL_EV;
						break;
					case PWRITE_EV:
						io_type = PWRITE_VAL_EV;
						break;
					case READV_EV:
						io_type = READV_VAL_EV;
						break;
					case WRITEV_EV:
						io_type = WRITEV_VAL_EV;
						break;
					case PREADV_EV:
						io_type = PREADV_VAL_EV;
						break;
					case PWRITEV_EV:
						io_type = PWRITEV_VAL_EV;
						break;
					default:
						io_type = 0;
						break;
				}
				trace_paraver_event (cpu, ptask, task, thread, time, IO_EV, io_type);
				trace_paraver_event (cpu, ptask, task, thread, time, IO_DESCRIPTOR_EV, EvParam);
				break;
			case EVT_BEGIN+1:
				/* This event refers to the size of the read/write operation */
				trace_paraver_event (cpu, ptask, task, thread, time, IO_SIZE_EV, EvParam);
				break;
			case EVT_BEGIN+2:
				/* This event refers to the type of file descriptor */
				trace_paraver_event (cpu, ptask, task, thread, time, IO_DESCRIPTOR_TYPE_EV, EvParam);
				break;
			case EVT_BEGIN+3:
				/* This event refers to the name of the file (only for open calls).
                                 * EvParam is the task's local file identifier. At this point we're in the 1st
                                 * stage of the merger, so we don't know how to translate the local id into 
                                 * an unified one because the translation information has not been shared yet. 
                                 * At the end of phase 1 the information is shared, and during phase 2 we will
                                 * change the local ids into the unifieds (see paraver_build_multi_event in paraver_generator.c)
                                 */
                                trace_paraver_event (cpu, ptask, task, thread, time, FILE_NAME_EV, EvParam);
                                break;
			default:
				break;
		}
	}
	else
		trace_paraver_event (cpu, ptask, task, thread, time, IO_EV, 0);

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
	task_t * task_info;
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
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType,
	  EvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time,
	  CLOCK_FROM_SYSTEM_EV, Get_EvTime (current_event));

	return 0;
}

/******************************************************************************
 ***  CPUEventInterval_Event
 ******************************************************************************/

static int CPUEventInterval_Event (event_t * current_event,
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

	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType,
	  EvValue);
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
	int EvType;
	unsigned long long EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvValue (current_event);     /* Value is the user event type.  */
	EvValue = Get_EvMiscParam (current_event); /* Param is the user event value. */

	/* Check whether we have to translate the events because they're registered
	   as callstack info */

	if (Extrae_Vector_Count (&RegisteredCodeLocationTypes) > 0)
	{
		unsigned u;
		unsigned umax = Extrae_Vector_Count (&RegisteredCodeLocationTypes);
		int found = FALSE;
		Extrae_Addr2Type_t *addr2types = NULL;

		for (u = 0; u < umax; u++)
		{
			addr2types = Extrae_Vector_Get (&RegisteredCodeLocationTypes, u);
			found = addr2types->LineType == EvType;
			/* Probably could be FunctionType also instead of LineType*/

			if (found)
				break;
		}

#if defined(HAVE_BFD)
		if (found && get_option_merge_SortAddresses() && EvValue != 0)
		{
			AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, addr2types->FunctionType_lbl);
			AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, addr2types->LineType_lbl);
		}
#endif

		trace_paraver_state (cpu, ptask, task, thread, current_time);
		if (found && addr2types != NULL)
		{
			trace_paraver_event (cpu, ptask, task, thread, current_time, addr2types->FunctionType, EvValue);
			trace_paraver_event (cpu, ptask, task, thread, current_time, addr2types->LineType, EvValue);
		}
		else
			trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
	}
	else
	{
		trace_paraver_state (cpu, ptask, task, thread, current_time);
		trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
	}

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
	thread_t *thread_info = GET_THREAD_INFO(ptask, task, thread);
	unsigned i, deepness;	
	UINT64 EvValue = Get_EvValue(current_event);
	UNREFERENCED_PARAMETER(fset);

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	deepness = Get_EvEvent(current_event) - CALLER_EV;
	if (deepness > 0 && deepness < MAX_CALLERS) 
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
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue,
		  ADDR2MPI_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue,
		  ADDR2MPI_LINE);
	}
#endif

	trace_paraver_event (cpu, ptask, task, thread, current_time,
	  CALLER_EV+deepness, EvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time,
	  CALLER_LINE_EV+deepness, EvValue);

	if (deepness > 0 && deepness < MAX_CALLERS)
		thread_info->AddressSpace_calleraddresses[deepness] = EvValue;

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
	unsigned int EvType;
	unsigned long long EvValue;
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
    unsigned int EvType;
	unsigned long long EvValue;
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
	unsigned int EvType;
	unsigned long long EvValue;
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
 ***  InitTracing_Event
 ******************************************************************************/
static int InitTracing_Event (event_t * current,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t *fset )
{
	UINT64 EvValue = Get_EvValue (current);

	UNREFERENCED_PARAMETER(fset);

	Switch_State (STATE_INITFINI, (EvValue == EVT_BEGIN), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, TRACE_INIT_EV, EvValue);

	if (EvValue == EVT_BEGIN)
	{
		UINT32 PID = Get_EvTarget (current);
		UINT32 PPID = Get_EvSize (current);
		UINT32 Depth = Get_EvTag (current);
		trace_paraver_event (cpu, ptask, task, thread, current_time, PID_EV, PID);
		trace_paraver_event (cpu, ptask, task, thread, current_time, PPID_EV, PPID);
		trace_paraver_event (cpu, ptask, task, thread, current_time, FORK_DEPTH_EV, Depth);
	}

	return 0;
}



/******************************************************************************
 ***  USRFunction_Event
 ******************************************************************************/
static int USRFunction_Event (event_t * current,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t *fset )
{
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvValue = Get_EvValue (current);

	/* HSG, I think this is not true... we should only maintain the previous
	   state Switch_State (STATE_RUNNING, (EvValue != EVT_END), ptask, task, thread);
	*/

#if defined(HAVE_BFD)
	if (get_option_merge_SortAddresses() && EvValue != 0)
	{
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2UF_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2UF_LINE);
	}
#endif

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, USRFUNC_EV, EvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time, USRFUNC_LINE_EV, EvValue);

	return 0;
}

/******************************************************************************
 ***  Sampling_Address_Event
 ******************************************************************************/
static int Sampling_Address_Event (event_t * current,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	uint64_t *CallerAddresses;
	unsigned i;
	int EvType;
	UINT64 EvValue;
	UINT64 EvParam;
	task_t *task_info = GET_TASK_INFO(ptask, task);
	UNREFERENCED_PARAMETER(fset);

	EvType = Get_EvEvent (current);
	EvValue = Get_EvValue (current);
	EvParam = Get_EvMiscParam (current); /* Param is the sampled address */

	if (Sample_Caller_Labels_Used == NULL) 
	{
		Sample_Caller_Labels_Used = (int *)malloc(sizeof(int)*MAX_CALLERS);
		for (i = 0; i < MAX_CALLERS; i++) 
			Sample_Caller_Labels_Used[i] = FALSE;
	}	     
	if (Sample_Caller_Labels_Used != NULL) 
		Sample_Caller_Labels_Used [0] = TRUE; 

	if (EvValue != 0)
	{

#if defined(HAVE_BFD)
		if (get_option_merge_SortAddresses())
		{
			AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2SAMPLE_FUNCTION);
			AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2SAMPLE_LINE);
		}
#endif

		/* HSG, samples should not break states?
		trace_paraver_state (cpu, ptask, task, thread, current_time);
		*/
		trace_paraver_event (cpu, ptask, task, thread, current_time, SAMPLING_EV, EvValue);
		trace_paraver_event (cpu, ptask, task, thread, current_time, SAMPLING_LINE_EV, EvValue);
	}

	if (EvParam != 0)
		trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvParam);

	if (AddressSpace_search (task_info->AddressSpace, EvParam, &CallerAddresses,
	    NULL))
	{
		unsigned u;
		for (u = 0; u < MAX_CALLERS; u++)
		{
			if (CallerAddresses[u] != 0)
			{
				trace_paraver_event (cpu, ptask, task, thread, current_time,
				  SAMPLING_ADDRESS_ALLOCATED_OBJECT_CALLER_EV+u,
				  CallerAddresses[u]);
			}
		}
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  SAMPLING_ADDRESS_ALLOCATED_OBJECT_EV, 0);
	}
	else
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  SAMPLING_ADDRESS_STATIC_OBJECT_EV, EvParam);

	return 0;
}

/******************************************************************************
 ***  Sampling_Address_MEM_TLB_Event
 ******************************************************************************/
static int Sampling_Address_MEM_TLB_Event (event_t * current,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	UINT64 EvValue;
	UINT64 EvParam;
	UNREFERENCED_PARAMETER(fset);

	EvValue = Get_EvValue (current);	 /* Value refers to hit or miss */
	EvParam = Get_EvMiscParam (current); /* Param refers to where data was obtained from */

	if (Get_EvEvent (current) == SAMPLING_ADDRESS_MEM_LEVEL_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  SAMPLING_ADDRESS_MEM_LEVEL_EV, EvParam);
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  SAMPLING_ADDRESS_MEM_HITORMISS_EV, EvValue);
	}
	else if (Get_EvEvent (current) == SAMPLING_ADDRESS_TLB_LEVEL_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  SAMPLING_ADDRESS_TLB_LEVEL_EV, EvParam);
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  SAMPLING_ADDRESS_TLB_HITORMISS_EV, EvValue);
	}
	else if (Get_EvEvent (current) == SAMPLING_ADDRESS_REFERENCE_COST_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  SAMPLING_ADDRESS_REFERENCE_COST_EV, EvValue);
	}

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
	unsigned EvTypeDelta, i;
	UINT64 EvValue;
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
				AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2SAMPLE_FUNCTION);
				AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2SAMPLE_LINE);
			}
			else
			{
				/* If depth != 0 (in EvTypeDelta), addresses are taken from the callstack
				   and point to the next instruction, so substract 1 */
				AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue-1, ADDR2SAMPLE_FUNCTION);
				AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue-1, ADDR2SAMPLE_LINE);
			}
		}
#endif

		/* HSG, samples should not break states?
		trace_paraver_state (cpu, ptask, task, thread, current_time);
		*/

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
	nthreads = (GET_TASK_INFO(ptask,task))->nthreads;
	for (i = 1; i <= nthreads; i++)
		HardwareCounters_NewSetDefinition(ptask, task, i, newSet, HWCIds);

	return 0;
}

/******************************************************************************
 **      Function name : ResetCounters
 **      Description :
 ******************************************************************************/

static void ResetCounters (unsigned ptask, unsigned task, unsigned thread)
{
	unsigned cnt;
	thread_t * Sthread = GET_THREAD_INFO(ptask, task, thread); 
	task_t *Stask = GET_TASK_INFO(ptask, task);

	Stask->tracing_disabled = FALSE;

	for (cnt = 0; cnt < MAX_HWC; cnt++)
		Sthread->counters[cnt] = 0;
}

/******************************************************************************
 **      Function name : HWC_Change_Ev
 **      Description :
 ******************************************************************************/

static int HWC_Change_Ev (
   event_t *current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset)
{
	int i;
	int hwctype[MAX_HWC+1];
	int prev_hwctype[MAX_HWC];
	unsigned long long hwcvalue[MAX_HWC+1];
	thread_t * Sthread;
	int oldSet = HardwareCounters_GetCurrentSet(ptask, task, thread);
	int *oldIds = HardwareCounters_GetSetIds(ptask, task, thread, oldSet);

	UNREFERENCED_PARAMETER(fset);

	Sthread = GET_THREAD_INFO(ptask, task, thread);
	Sthread->last_hw_group_change = current_time;
	Sthread->HWCChange_count++;
	int newSet = Get_EvValue(current_event);

	/* HSG changing the HWC set do not should change the application state */
	/* trace_paraver_state (cpu, ptask, task, thread, current_time); */

	/* Store which were the counters being read before (they're overwritten with the new set at HardwareCounters_Change) */
	for (i=0; i<MAX_HWC; i++)
	{
#if defined(PMAPI_COUNTERS)
		prev_hwctype[i] = HWC_COUNTER_TYPE(i, oldsIds[i]);
#else
		prev_hwctype[i] = HWC_COUNTER_TYPE(oldIds[i]);
#endif
	}

	ResetCounters (ptask, task, thread);
	HardwareCounters_Change (ptask, task, thread, newSet, hwctype, hwcvalue);

	/* This loop starts at 0 and goes to MAX_HWC+1 because HardwareCounters_Change
	   reports in hwctype[0] the counter group identifier */
	for (i = 0; i < MAX_HWC+1; i++)
	{
		if (NO_COUNTER != hwctype[i] && Sthread->HWCChange_count > 1)
		{
			int found = FALSE, k = 0;

			/* Check the current counter (hwctype[i]) did not appear on the previous set. We don't want
			 * it to appear twice in the same timestamp. This may happen because the HWC_CHANGE_EV is traced
			 * right after the last valid emission of counters with the previous set, at the same timestamp.
			 */

			while (!found && k < MAX_HWC)
			{
				if (hwctype[i] == prev_hwctype[k])
					found = TRUE;
				k ++;
			}

			if (!found)
			{
				trace_paraver_event (cpu, ptask, task, thread, current_time, hwctype[i], hwcvalue[i]);
			}
		}
		/*
		 * The first time we read the counters we cannot rely on their value, so
		 * we set them to 0.
		 */
		else if (NO_COUNTER != hwctype[i] && Sthread->HWCChange_count == 1)
		{
			if (i > 0)
			{
				trace_paraver_event (cpu, ptask, task, thread, current_time, hwctype[i], 0);
			}
			else
			{
				/* Index [0] contains the active set, not a counter. We always have to
				 * emit its actual value.
				 */
				trace_paraver_event (cpu, ptask, task, thread, current_time, hwctype[0], hwcvalue[0]);
			}
		}
	}
	return 0;
}

#if defined(DEAD_CODE)
static int Evt_SetCounters (
   event_t * current_event,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread,
   FileSet_t *fset )
{
	thread_t * Sthread;

	UNREFERENCED_PARAMETER(fset);
	unsigned int newSet = Get_EvValue (current_event);

	Sthread = GET_THREAD_INFO(ptask, task, thread);
	Sthread->last_hw_group_change = current_time;

	return HWC_Change_Ev (newSet, current_time, cpu, ptask, task, thread);
}
#endif

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
	UNREFERENCED_PARAMETER(fset);
	UINT64 EvValue;

	if (Get_EvEvent( current_event ) == ONLINE_EV)
	{
		EvValue = Get_EvMiscParam (current_event);
	}
	else
	{
		EvValue = Get_EvValue (current_event);
	}

	Switch_State (STATE_RUNNING, (EvValue == EVT_BEGIN), ptask, task, thread);
	trace_paraver_state (cpu, ptask, task, thread, current_time);

	/* DEBUG -- we don't trace this event in CPU Burst mode. This is just for debugging purposes
	trace_paraver_event (cpu, ptask, task, thread, current_time, CPU_BURST_EV, EvValue); */

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

static int Online_Event (event_t * current_event,
    unsigned long long current_time, unsigned int cpu, unsigned int ptask,
    unsigned int task, unsigned int thread, FileSet_t *fset)
{
  unsigned int EvType, EvValue;
  UNREFERENCED_PARAMETER(fset);

  EvType  = Get_EvValue (current_event);
  EvValue = Get_EvMiscParam (current_event);

  switch(EvType)
  {
    case RAW_PERIODICITY_EV:
    case RAW_BEST_ITERS_EV:
    case PERIODICITY_EV:
      HaveSpectralEvents = TRUE;
      MaxRepresentativePeriod = MAX(MaxRepresentativePeriod, EvValue);
      trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
      break;

    case DETAIL_LEVEL_EV:
      HaveSpectralEvents = TRUE;
      /* Remove any unclosed state */
//      Pop_Until (STATE_RUNNING, ptask, task, thread);

      if (EvValue != DETAIL_MODE)
      {
        /* Clear pending unmatched communications so that they don't match when the tracing is restarted */

        MatchComms_Off (ptask, task);
      }

      if (EvValue == DETAIL_MODE)
      {
        Initialize_Trace_Mode_States( cpu, ptask, task, thread, TRACE_MODE_DETAIL );
      }
      if (EvValue == BURST_MODE)
      {
        Initialize_Trace_Mode_States( cpu, ptask, task, thread, TRACE_MODE_BURSTS );
      }
      if (EvValue == PHASE_PROFILE)
      {
        Initialize_Trace_Mode_States( cpu, ptask, task, thread, TRACE_MODE_PHASE_PROFILE );
      } 
      if (EvValue == NOT_TRACING)
      {
        Initialize_Trace_Mode_States( cpu, ptask, task, thread, TRACE_MODE_DISABLED );
      }

      trace_paraver_state (cpu, ptask, task, thread, current_time);
      trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
      break;

    case CLUSTER_ID_EV:
      MaxClusterId = MAX(MaxClusterId, EvValue);
      trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
      break;

    case CLUSTER_SUPPORT_EV:
      trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
      break;

    case ONLINE_STATE_EV:
      Switch_State( STATE_ONLINE_ANALYSIS, (EvValue == ONLINE_PAUSE_APP), ptask, task, thread);
      trace_paraver_state (cpu, ptask, task, thread, current_time);
      trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
      break;

    case MPI_STATS_P2P_COUNT_EV:
    case MPI_STATS_P2P_BYTES_SENT_EV:
    case MPI_STATS_P2P_BYTES_RECV_EV:
    case MPI_STATS_GLOBAL_COUNT_EV:
    case MPI_STATS_GLOBAL_BYTES_SENT_EV:
    case MPI_STATS_GLOBAL_BYTES_RECV_EV:
    case MPI_STATS_TIME_IN_MPI_EV:
    case MPI_STATS_P2P_INCOMING_COUNT_EV:
    case MPI_STATS_P2P_OUTGOING_COUNT_EV:
    case MPI_STATS_P2P_INCOMING_PARTNERS_COUNT_EV:
    case MPI_STATS_P2P_OUTGOING_PARTNERS_COUNT_EV:
    case MPI_STATS_TIME_IN_OTHER_EV:
    case MPI_STATS_TIME_IN_P2P_EV:
    case MPI_STATS_TIME_IN_GLOBAL_EV:
    case MPI_STATS_OTHER_COUNT_EV:
      MPI_Stats_Event( current_event, current_time, cpu, ptask, task, thread, fset );
      break;

    case CPU_BURST_EV:
//      fprintf(stderr, "[DEBUG] ONLINE_EV: CPU_BURST_EV: Val=%d\n", EvValue);
      CPU_Burst_Event( current_event, current_time, cpu, ptask, task, thread, fset );
      break;

    case GREMLIN_EV:
      trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);
      break;

  }
  return 0;
}

#if 0
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
#endif

/******************************************************************************
 ***  User_Send_Event
 ******************************************************************************/

static int User_Send_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned partner, recv_thread, recv_vthread;
	task_t *task_info, *task_info_partner;
	thread_t *thread_info;
	event_t * recv_begin, * recv_end;
	UNREFERENCED_PARAMETER(cpu);

	task_info = GET_TASK_INFO(ptask, task);
	thread_info = GET_THREAD_INFO(ptask, task, thread);

	if (MatchComms_Enabled(ptask, task))
	{
		if (Get_EvTarget(current_event)==EXTRAE_COMM_PARTNER_MYSELF)
			partner = task-1;
		else
			partner = Get_EvTarget(current_event);

		if (isTaskInMyGroup (fset, ptask-1, partner))
		{
			task_info_partner = GET_TASK_INFO(ptask, partner+1);

#if defined(DEBUG)
			fprintf (stdout, "USER SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", Get_EvEvent(current_event), current_time, Get_EvTime(current_event), task-1, partner, Get_EvTag(current_event));
#endif

			CommunicationQueues_ExtractRecv (task_info_partner->recv_queue, task-1, Get_EvTag (current_event), &recv_begin, &recv_end, &recv_thread, &recv_vthread, Get_EvAux(current_event));

			if (recv_begin == NULL || recv_end == NULL)
			{
				off_t position;

#if defined(DEBUG)
				fprintf (stdout, "USER SEND_CMD(%u) DID NOT find receiver\n", Get_EvEvent(current_event));
#endif

				position = WriteFileBuffer_getPosition (thread_info->file->wfb);
				CommunicationQueues_QueueSend (task_info->send_queue, current_event, current_event, position, thread, thread_info->virtual_thread, partner, Get_EvTag(current_event), Get_EvAux(current_event));
				trace_paraver_unmatched_communication (1, ptask, task, thread, thread_info->virtual_thread, current_time, Get_EvTime(current_event), 1, ptask, partner+1, recv_thread, Get_EvSize(current_event), Get_EvTag(current_event));
			}
			else
			{
				
#if defined(DEBUG)
				fprintf (stdout, "USER SEND_CMD(%u) DID NOT find receiver\n", Get_EvEvent(current_event));
#endif
				trace_communicationAt (ptask, task, thread, thread_info->virtual_thread, ptask, partner+1, recv_thread, recv_vthread, current_event, current_event, recv_begin, recv_end, FALSE, 0);
			}
		}
#if defined(PARALLEL_MERGE)
		else
		{
#if defined(DEBUG)
			fprintf (stdout, "SEND_CMD(%u): TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d >> PENDING\n", Get_EvEvent(current_event), current_time, Get_EvTime(current_event), task-1, Get_EvTarget(current_event), Get_EvTag(current_event));
#endif
			trace_pending_communication (ptask, task, thread, thread_info->virtual_thread, current_event, current_event, ptask, partner);
		}
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
	unsigned partner, send_thread, send_vthread;
	task_t *task_info, *task_info_partner;
	thread_t *thread_info;
	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(current_time);

	task_info = GET_TASK_INFO(ptask, task);
	thread_info = GET_THREAD_INFO(ptask, task, thread);

	if (MatchComms_Enabled(ptask, task))
	{
		if (Get_EvTarget(current_event)==EXTRAE_COMM_PARTNER_MYSELF)
			partner = task-1;
		else
			partner = Get_EvTarget(current_event);

		if (isTaskInMyGroup (fset, ptask-1, partner))
		{
			task_info_partner = GET_TASK_INFO(ptask, partner+1);

#if defined(DEBUG)
			fprintf (stdout, "USER RECV_CMD: TIME/TIMESTAMP %lld/%lld IAM %d PARTNER %d tag %d\n", current_time, Get_EvTime(current_event), task-1, partner, Get_EvTag(current_event));
#endif

			CommunicationQueues_ExtractSend (task_info_partner->send_queue, task-1, Get_EvTag (current_event), &send_begin, &send_end, &send_position, &send_thread, &send_vthread, Get_EvAux(current_event));

			if (NULL == send_begin || NULL == send_end)
			{
#if defined(DEBUG)
						fprintf (stdout, "USER RECV_CMD DID NOT find partner\n");
#endif
				CommunicationQueues_QueueRecv (task_info->recv_queue, current_event, current_event, thread, thread_info->virtual_thread, partner, Get_EvTag(current_event), Get_EvAux(current_event));
			}
			else if (NULL != send_begin && NULL != send_end)
			{
#if defined(DEBUG)
						fprintf (stdout, "USER RECV_CMD find partner\n");
#endif
				trace_communicationAt (ptask, partner+1, send_thread, send_vthread, ptask, task, thread, thread_info->virtual_thread, send_begin, send_end, current_event, current_event, TRUE, send_position);
			}
			else
				fprintf (stderr, "mpi2prv: Attention CommunicationQueues_ExtractSend returned send_begin = %p and send_end = %p\n", send_begin, send_end);
		}
#if defined(PARALLEL_MERGE)
		else
		{
			UINT64 log_r, phy_r;

			log_r = TIMESYNC (ptask-1, task-1, Get_EvTime(current_event));
			phy_r = TIMESYNC (ptask-1, task-1, Get_EvTime(current_event));
			AddForeignRecv (phy_r, log_r, Get_EvTag(current_event), ptask-1, task-1, thread-1,
			  thread_info->virtual_thread-1, ptask-1, partner, fset, MatchComms_GetZone(ptask, task));
		}
#endif
	}

	return 0;
}

/******************************************************************************
 ***  Resume_Virtual_Thread_Event
 ******************************************************************************/

static int Resume_Virtual_Thread_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	thread_t *thread_info = GET_THREAD_INFO(ptask, task, thread);
	task_t *task_info = GET_TASK_INFO(ptask, task);

	UNREFERENCED_PARAMETER(fset);

	if (!get_option_merge_NanosTaskView())
	{
		unsigned i, u;
		unsigned new_active_task_thread = Get_EvValue(current_event);

		/* If this is a new virtual thread, allocate its information within the TASK */
		if (task_info->num_active_task_threads < new_active_task_thread)
		{
			/* Allocate memory for the new coming threads */
			task_info->active_task_threads = (active_task_thread_t*) realloc (task_info->active_task_threads,
			  new_active_task_thread*sizeof(active_task_thread_t));
			if (task_info->active_task_threads == NULL)
			{
				fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate information for active task threads\n");
				exit (0);
			}

			/* Init their structures */
			for (u = task_info->num_active_task_threads; u < new_active_task_thread; u++)
			{
				task_info->active_task_threads[u].stacked_type = NULL;
				task_info->active_task_threads[u].num_stacks = 0;
			}

			task_info->num_active_task_threads = new_active_task_thread;
			thread_info->active_task_thread = new_active_task_thread;
		}
		else
		{
			/* Write as many "PUSHEs" in tracefile according to the current
			   stack, for each of the visited stacked types */
			active_task_thread_t *att = &(task_info->active_task_threads[new_active_task_thread-1]);
			for (u = 0; u < att->num_stacks; u++)
				for (i = 0; i < Stack_Depth(att->stacked_type[u].stack); i++)
					trace_paraver_event (cpu, ptask, task, thread, current_time,
					  att->stacked_type[u].type,
					  Stack_ValueAt(att->stacked_type[u].stack, i));

			thread_info->active_task_thread = new_active_task_thread;
		}
	}
	else
	{
		unsigned new_virtual_thread = Get_EvValue(current_event);

		thread_info->virtual_thread = new_virtual_thread;
		task_info->num_virtual_threads = MAX(new_virtual_thread, task_info->num_virtual_threads);
	}

//	trace_paraver_event (cpu, ptask, task, thread, current_time, 123456, Get_EvValue(current_event));

	return 0;
}

/******************************************************************************
 ***  Suspend_Virtual_Thread_Event
 ******************************************************************************/

static int Suspend_Virtual_Thread_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(current_event);
	UNREFERENCED_PARAMETER(fset);

	/* If we don't want to see nanos tasks, we need to support emit the
	   stacked values pops at suspend time */
	if (!get_option_merge_NanosTaskView())
	{
		unsigned u, i;
		thread_t *thread_info = GET_THREAD_INFO(ptask, task, thread);
		task_t *task_info = GET_TASK_INFO(ptask, task);
		active_task_thread_t *att = &(task_info->active_task_threads[thread_info->active_task_thread-1]);

		/* Write as many "POPs" (0s) in tracefile according to the current
		   stack depth, for each of the visited stacked types */
		for (u = 0; u < att->num_stacks; u++)
			for (i = 0; i < Stack_Depth(att->stacked_type[u].stack); i++)
				trace_paraver_event (cpu, ptask, task, thread, current_time,
				  att->stacked_type[u].type, 0);

//		trace_paraver_event (cpu, ptask, task, thread, current_time, 123456, 0);
	}

	return 0;
}

/******************************************************************************
 ***  Register_Stacked_Type_Event
 ******************************************************************************/

static int Register_Stacked_Type_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(thread);
	UNREFERENCED_PARAMETER(fset);

	if (!Vector_Search (RegisteredStackValues, Get_EvValue(current_event)))
		Vector_Add (RegisteredStackValues, Get_EvValue(current_event));

	return 0;
}


/******************************************************************************
 ***  Register_CodeLocation_Type_Event
 ******************************************************************************/

static int Register_CodeLocation_Type_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	int EvFunction;
	int EvLine;
	Extrae_Addr2Type_t *cl_types;

	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(thread);
	UNREFERENCED_PARAMETER(fset);

	EvFunction = Get_EvValue (current_event); /* Value refers to the function type  */
	EvLine = Get_EvMiscParam (current_event); /* Param refers to the file and line no */

	cl_types = Extrae_Addr2Type_New (EvFunction, ADDR2OTHERS_FUNCTION,
		EvLine, ADDR2OTHERS_LINE);

	if (!Extrae_Vector_Search (&RegisteredCodeLocationTypes, cl_types,
	     Extrae_Addr2Type_Compare))
	{
		Extrae_Vector_Append (&RegisteredCodeLocationTypes, cl_types);
	}

	return 0;
}

/******************************************************************************
 ***  Fork_Event
 ******************************************************************************/

static int ForkWaitSystem_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int EvType, EvValue;
	unsigned int state = 0;
	UNREFERENCED_PARAMETER(fset);

	switch (Get_EvEvent (current_event))
	{
		case SYSTEM_EV:
		case FORK_EV:
			state = STATE_OVHD;
			break;
		case WAIT_EV:
		case WAITPID_EV:
			state = STATE_BLOCKED;
			break;
		default:
			break;
	}

	Switch_State (state, (Get_EvValue(current_event) == EVT_BEGIN), ptask, task,
	  thread);

	if (Get_EvValue (current_event) == EVT_BEGIN)
		EvValue = MISC_event_GetValueForForkRelated (Get_EvEvent (current_event));
	else
		EvValue = 0;

	EvType = FORK_SYSCALL_EV;

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

/******************************************************************************
 ***  Exec_Event
 ******************************************************************************/

static int Exec_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(current_event);
	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(cpu);
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(thread);
	UNREFERENCED_PARAMETER(fset);

	return 0;
}

/******************************************************************************
 ***  GetCPU_Event
 ******************************************************************************/

static int GetCPU_Event (event_t * current_event,
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	UNREFERENCED_PARAMETER(fset);

	trace_paraver_event (cpu, ptask, task, thread, current_time,
	  Get_EvEvent (current_event), Get_EvValue (current_event));

	return 0;
}


/******************************************************************************
 ***  DynamicMemory_Event
 ******************************************************************************/

static int DynamicMemory_Event (event_t * event,
	unsigned long long time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset)
{
	task_t *task_info = GET_TASK_INFO(ptask, task);
	thread_t *thread_info = GET_THREAD_INFO(ptask, task, thread);
	UNREFERENCED_PARAMETER(fset);

	unsigned EvType = Get_EvEvent (event);
	unsigned long long EvParam = Get_EvParam (event);
	unsigned long long EvValue = Get_EvValue (event);
	int isBegin = EvValue == EVT_BEGIN;

	if ((EvType == MALLOC_EV)                 ||
	    (EvType == MEMKIND_MALLOC_EV)         ||
	    (EvType == MEMKIND_POSIX_MEMALIGN_EV) ||
	    (EvType == POSIX_MEMALIGN_EV)         ||
	    (EvType == KMPC_MALLOC_EV)            ||
	    (EvType == KMPC_ALIGNED_MALLOC_EV))
	{
		/* Malloc: in size, out pointer */
		if (isBegin)
		{
			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_REQUESTED_SIZE_EV, EvParam);

			/* Store size and time of creation for later use in isEnd */
			thread_info->AddressSpace_size = EvParam;
			thread_info->AddressSpace_timeCreation = time;
		}
		else
		{
			/* Emit information regarding the calling site to identify
			   where the structure was allocated */
			unsigned u;
			for (u = 0; u < MAX_CALLERS; u++)
			{
				if (thread_info->AddressSpace_calleraddresses[u] != 0)
					trace_paraver_event (cpu, ptask, task, thread,
					  thread_info->AddressSpace_timeCreation,
					  SAMPLING_ADDRESS_ALLOCATED_OBJECT_CALLER_EV+u,
					  thread_info->AddressSpace_calleraddresses[u]);
			}

			/* Emit event that will be replaced by the ID of the 
			   location of structure allocation */
			trace_paraver_event (cpu, ptask, task, thread,
			  thread_info->AddressSpace_timeCreation,
			  SAMPLING_ADDRESS_ALLOCATED_OBJECT_ALLOC_EV, 0);

			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_POINTER_OUT_EV, EvParam);

			AddressSpace_add (task_info->AddressSpace, EvParam,
			  EvParam+thread_info->AddressSpace_size,
			  thread_info->AddressSpace_calleraddresses,
			  thread_info->AddressSpace_callertype);
		}
	}
	else if ((EvType == FREE_EV)         ||
	         (EvType == MEMKIND_FREE_EV) ||
	         (EvType == KMPC_FREE_EV))
	{
		/* Free: in pointer */
		if (isBegin)
		{
			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_POINTER_IN_EV, EvParam);

			AddressSpace_remove (task_info->AddressSpace, EvParam);
		}
	}
	else if ((EvType == REALLOC_EV)         ||
	         (EvType == MEMKIND_REALLOC_EV) ||
	         (EvType == KMPC_REALLOC_EV))
	{
		/* Realloc: in size, in pointer (in EVT_BEGIN+1), out ptr*/
		if (EvValue == EVT_BEGIN)
		{
			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_POINTER_IN_EV, EvParam);

			thread_info->AddressSpace_size = EvParam;
		}
		else if (EvValue == EVT_BEGIN+1)
		{
			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_REQUESTED_SIZE_EV, EvParam);

			AddressSpace_remove (task_info->AddressSpace, EvParam);
		}
		else
		{
			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_POINTER_OUT_EV, EvParam);

			AddressSpace_add (task_info->AddressSpace, EvParam,
			  EvParam+thread_info->AddressSpace_size,
			  thread_info->AddressSpace_calleraddresses,
			  thread_info->AddressSpace_callertype);
		}
	
	}
	else if ((EvType == CALLOC_EV)         ||
	         (EvType == MEMKIND_CALLOC_EV) ||
	         (EvType == KMPC_CALLOC_EV))
	{
		/* Calloc: in size, out pointer */
		if (isBegin)
		{
			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_REQUESTED_SIZE_EV, EvParam);

			thread_info->AddressSpace_size = EvParam;
		}
		else
		{
			trace_paraver_event (cpu, ptask, task, thread, time,
			  DYNAMIC_MEM_POINTER_OUT_EV, EvParam);

			AddressSpace_add (task_info->AddressSpace, EvParam,
			  EvParam+thread_info->AddressSpace_size,
			  thread_info->AddressSpace_calleraddresses,
			  thread_info->AddressSpace_callertype);
		}
	}

	if (EvValue == EVT_BEGIN || EvValue == EVT_END)
	{
		// Do not change the state in MALLOC related calls
		// Switch_State (STATE_OTHERS, EvValue == EVT_BEGIN, ptask, task, thread);
		// trace_paraver_state (cpu, ptask, task, thread, time);

		unsigned PRVValue = isBegin?MISC_event_GetValueForDynamicMemory(EvType):0;
		Switch_State (Get_State(EvType), (EvValue == EVT_BEGIN), ptask, task, thread);
		trace_paraver_state (cpu, ptask, task, thread, time);

		trace_paraver_event (cpu, ptask, task, thread, time, DYNAMIC_MEM_EV, PRVValue);
	}

	if (!isBegin)
	{
		unsigned u;
		for (u = 0; u < MAX_CALLERS; u++)
			thread_info->AddressSpace_calleraddresses[u] = 0;
	}

	return 0;
}

static int DynamicMemory_Partition_Event (event_t * event,
        unsigned long long time, unsigned int cpu, unsigned int ptask,
        unsigned int task, unsigned int thread, FileSet_t *fset)
{
        unsigned EvType = Get_EvEvent (event);
        unsigned long long EvValue = Get_EvValue (event);

	trace_paraver_event (cpu, ptask, task, thread, time, MEMKIND_PARTITION_EV, EvValue);
}

static int SystemCall_Event (event_t * event,                      
	unsigned long long time, unsigned int cpu, unsigned int ptask,          
	unsigned int task, unsigned int thread, FileSet_t *fset)                
{                                                                               
	int i = 0;
	unsigned EvType = Get_EvEvent (event);                                  
	unsigned long long EvValue = Get_EvValue (event);
	unsigned long long SysCallID = Get_EvMiscParam (event);

  if (!Syscall_Events_Found)                                                  
	{                                                                             
	  Syscall_Events_Found = TRUE;                                              
		for (i=0; i<SYSCALL_EVENTS_COUNT; i++)                                    
		{                                                                           
			Syscall_Labels_Used[i] = FALSE;                                         
		}                                                                           
  }                                                                             
	Syscall_Labels_Used[SysCallID] = TRUE;

	trace_paraver_event (cpu, ptask, task, thread, time, 
	 SYSCALL_EV, (EvValue == EVT_BEGIN ? SysCallID+1 : 0));
}                                                                               


/*****************************************************************************/

SingleEv_Handler_t PRV_MISC_Event_Handlers[] = {
	{ FLUSH_EV, Flush_Event },
        { OPEN_EV, ReadWrite_Event },
        { FOPEN_EV, ReadWrite_Event },
	{ READ_EV, ReadWrite_Event },
	{ WRITE_EV, ReadWrite_Event },
	{ FREAD_EV, ReadWrite_Event },
	{ FWRITE_EV, ReadWrite_Event },
	{ PREAD_EV, ReadWrite_Event },
	{ PWRITE_EV, ReadWrite_Event },
	{ READV_EV, ReadWrite_Event },
	{ PREADV_EV, ReadWrite_Event },
	{ WRITEV_EV, ReadWrite_Event },
	{ PWRITEV_EV, ReadWrite_Event },
	{ APPL_EV, Appl_Event },
	{ TRACE_INIT_EV, InitTracing_Event },
	{ USER_EV, User_Event },
	{ HWC_EV, SkipHandler }, /* Automatically done outside */
#if USE_HARDWARE_COUNTERS
	{ HWC_DEF_EV, Evt_CountersDefinition },
	{ HWC_CHANGE_EV, HWC_Change_Ev },
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
	{ USRFUNC_EV, USRFunction_Event },
	{ TRACING_MODE_EV, Tracing_Mode_Event },
	{ ONLINE_EV, Online_Event },
	{ USER_SEND_EV, User_Send_Event },
	{ USER_RECV_EV, User_Recv_Event },
	{ RESUME_VIRTUAL_THREAD_EV, Resume_Virtual_Thread_Event },
	{ SUSPEND_VIRTUAL_THREAD_EV, Suspend_Virtual_Thread_Event },
	{ REGISTER_STACKED_TYPE_EV, Register_Stacked_Type_Event },
	{ REGISTER_CODELOCATION_TYPE_EV, Register_CodeLocation_Type_Event },
	{ FORK_EV, ForkWaitSystem_Event },
	{ WAIT_EV, ForkWaitSystem_Event },
	{ SYSTEM_EV, ForkWaitSystem_Event },
	{ WAITPID_EV, ForkWaitSystem_Event },
	{ EXEC_EV, Exec_Event },
	{ GETCPU_EV, GetCPU_Event },
	{ CPU_EVENT_INTERVAL_EV, CPUEventInterval_Event },
	{ SAMPLING_ADDRESS_LD_EV, Sampling_Address_Event },
	{ SAMPLING_ADDRESS_ST_EV, Sampling_Address_Event },
	{ SAMPLING_ADDRESS_MEM_LEVEL_EV, Sampling_Address_MEM_TLB_Event },
	{ SAMPLING_ADDRESS_TLB_LEVEL_EV, Sampling_Address_MEM_TLB_Event },
	{ SAMPLING_ADDRESS_REFERENCE_COST_EV, Sampling_Address_MEM_TLB_Event },
	{ MALLOC_EV, DynamicMemory_Event },
	{ CALLOC_EV, DynamicMemory_Event },
	{ FREE_EV, DynamicMemory_Event },
	{ REALLOC_EV, DynamicMemory_Event },
	{ POSIX_MEMALIGN_EV, DynamicMemory_Event },
	{ MEMKIND_MALLOC_EV, DynamicMemory_Event },
	{ MEMKIND_CALLOC_EV, DynamicMemory_Event },
	{ MEMKIND_REALLOC_EV, DynamicMemory_Event },
	{ MEMKIND_POSIX_MEMALIGN_EV, DynamicMemory_Event },
	{ MEMKIND_FREE_EV, DynamicMemory_Event },
	{ MEMKIND_PARTITION_EV, DynamicMemory_Partition_Event },
	{ SYSCALL_EV, SystemCall_Event },
	{ KMPC_MALLOC_EV, DynamicMemory_Event },
	{ KMPC_CALLOC_EV, DynamicMemory_Event },
	{ KMPC_FREE_EV, DynamicMemory_Event },
	{ KMPC_REALLOC_EV, DynamicMemory_Event },
	{ KMPC_ALIGNED_MALLOC_EV, DynamicMemory_Event },
	{ NULL_EV, NULL }
};

RangeEv_Handler_t PRV_MISC_Range_Handlers[] = {
	{ CALLER_EV, CALLER_EV + MAX_CALLERS, MPI_Caller_Event },
	{ SAMPLING_EV, SAMPLING_EV + MAX_CALLERS, Sampling_Caller_Event },
	{ NULL_EV, NULL_EV, NULL }
};

