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

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# ifdef HAVE_FOPEN64
#  define __USE_LARGEFILE64
# endif
# include <stdio.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#if defined (PARALLEL_MERGE)
# include "mpi.h"
# include "mpi-tags.h"
# include "mpi-aux.h"
#endif

#include "file_set.h"
#include "events.h"
#include "record.h"
#include "object_tree.h"
#include "mpi2out.h"
#include "trace_to_prv.h"
#include "mpi_prv_semantics.h"
#include "misc_prv_semantics.h"
#include "omp_prv_semantics.h"
#include "pthread_prv_semantics.h"
#include "mpi_comunicadors.h"
#include "labels.h"
#include "trace_mode.h"
#include "semantics.h"
#include "dump.h"
#include "paraver_generator.h"
#include "paraver_state.h"
#include "options.h"

#include "mpi_prv_events.h"
#include "addr2info.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
# include "tree-logistics.h"
#endif

//#define DEBUG

/* Variable will hold whether all times end with 000 even if the timing routine
   is told to be generating nanoseconds! */
static int TimeIn_MicroSecs = TRUE;

#define CHECK_TIME_US(x) \
	{ TimeIn_MicroSecs = TimeIn_MicroSecs && ((x%1000)==0); }

/******************************************************************************
 ***  PRVWRITECNTL
 ***  Macro per controlar si hi ha un error d'escriptura al fitxer .prv
 ******************************************************************************/
#define PRVWRITECNTL(a) \
  if ((a)<0) \
  { \
    fprintf(stderr,"mpi2prv ERROR : Writing to disk the tracefile\n"); \
    return -1; \
  }

/******************************************************************************
 ***  trace_paraver_record & trace_paraver_recordAt
 ******************************************************************************/
static void trace_paraver_record (WriteFileBuffer_t *wfb,
	paraver_rec_t *record)
{
	WriteFileBuffer_write (wfb, record);
}

static void trace_paraver_recordAt (WriteFileBuffer_t *wfb,
	paraver_rec_t *record, off_t position)
{
	WriteFileBuffer_writeAt (wfb, record, position);
}

/******************************************************************************
 ***  trace_paraver_state_noahead
 ******************************************************************************/
void trace_paraver_state_noahead (
   unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
   unsigned long long current_time)
{
	thread_t * thread_info = GET_THREAD_INFO (ptask, task, thread);
	WriteFileBuffer_t *wfb = thread_info->file->wfb;
	unsigned current_state = Top_State(ptask, task, thread);

	UNREFERENCED_PARAMETER(cpu);

#if 0
	fprintf (stderr, "trace_paraver_state (..)\n");
	fprintf (stderr, "thread_info->incomplete_state_offset = %u\n", thread_info->incomplete_state_offset);
#endif

	/* Complete the previous state */
	if (thread_info->incomplete_state_offset != (off_t)-1) /* This isn't the first state */
	{
#if 0
		fprintf (stderr, "get_option_merge_JointStates() = %d Get_Last_State() = %d\n", get_option_merge_JointStates(), Get_Last_State());
		fprintf (stderr, "thread_info->incomplete_state_record.value = %d == current_state = %d\n", thread_info->incomplete_state_record.value, current_state);
#endif

		/* Do not split states whether appropriate */
		if (get_option_merge_JointStates() && !Get_Last_State())
			if (thread_info->incomplete_state_record.value == current_state)
				return;

		/* Write the record into the *.tmp file if the state isn't excluded */
#if defined(DEBUG_STATES)
		fprintf(stderr, "mpi2prv: DEBUG [T:%d] Closing state %u at %llu\n", task,  
		(unsigned int)thread_info->incomplete_state_record.value, current_time);
		fprintf (stderr, "Excluded? %d\n", State_Excluded(thread_info->incomplete_state_record.value));
#endif

		if (!State_Excluded(thread_info->incomplete_state_record.value))
		{
			thread_info->incomplete_state_record.end_time = current_time;
			WriteFileBuffer_writeAt (wfb, &(thread_info->incomplete_state_record), thread_info->incomplete_state_offset);
		}
	}
}

/******************************************************************************
 ***  trace_paraver_state
 ******************************************************************************/
void trace_paraver_state (
   unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
   unsigned long long current_time)
{
	thread_t * thread_info = GET_THREAD_INFO (ptask, task, thread);
	WriteFileBuffer_t *wfb = thread_info->file->wfb;
	unsigned current_state = Top_State(ptask, task, thread);

#if 0
	fprintf (stderr, "trace_paraver_state (..)\n");
	fprintf (stderr, "thread_info->incomplete_state_offset = %u\n", thread_info->incomplete_state_offset);
#endif

	/* Complete the previous state */
	if (thread_info->incomplete_state_offset != (off_t)-1) /* This isn't the first state */
	{
#if 0
		fprintf (stderr, "get_option_merge_JointStates() = %d Get_Last_State() = %d\n", get_option_merge_JointStates(), Get_Last_State());
		fprintf (stderr, "thread_info->incomplete_state_record.value = %d == current_state = %d\n", thread_info->incomplete_state_record.value, current_state);
#endif

		/* Do not split states whether appropriate */
		if (get_option_merge_JointStates() && !Get_Last_State())
			if (thread_info->incomplete_state_record.value == current_state)
				return;

		/* Write the record into the *.tmp file if the state isn't excluded */
#if defined(DEBUG_STATES)
		fprintf(stderr, "mpi2prv: DEBUG [T:%d] Closing state %u at %llu\n", task,  
		(unsigned int)thread_info->incomplete_state_record.value, current_time);
		fprintf (stderr, "Excluded? %d\n", State_Excluded(thread_info->incomplete_state_record.value));
#endif

		if (!State_Excluded(thread_info->incomplete_state_record.value))
		{
			thread_info->incomplete_state_record.end_time = current_time;
			WriteFileBuffer_writeAt (wfb, &(thread_info->incomplete_state_record), thread_info->incomplete_state_offset);
		}
	}

	/* Start the next state */
	thread_info->incomplete_state_record.type   = STATE;
	thread_info->incomplete_state_record.cpu    = cpu;
	thread_info->incomplete_state_record.ptask  = ptask;
	thread_info->incomplete_state_record.task   = task;
	thread_info->incomplete_state_record.thread = thread_info->virtual_thread;
	thread_info->incomplete_state_record.time   = current_time;
	thread_info->incomplete_state_record.value  = current_state;
	/* Save a slot in the *.tmp file for this record if this state isn't excluded */
#if defined(DEBUG_STATES)
	fprintf(stderr, "mpi2prv: DEBUG [T:%d] Starting state %u (excluded? %d) at %llu\n", task, current_state, State_Excluded(current_state), current_time);
#endif
	if (!State_Excluded(current_state))
	{
		paraver_rec_t fake_record;
		fake_record.type   = UNFINISHED_STATE;
		fake_record.ptask  = ptask;
		fake_record.task   = task;
		fake_record.thread = thread;
		fake_record.time   = current_time;
		thread_info->incomplete_state_offset = WriteFileBuffer_getPosition (wfb);
		trace_paraver_record (wfb, &fake_record);
	}
}


/******************************************************************************
 ***  trace_paraver_event
 ******************************************************************************/
void trace_paraver_event (
   unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
   unsigned long long time, 
   unsigned int type, UINT64 value)
{
	thread_t * thread_info;
	paraver_rec_t record;
	int tipus;
	UINT64 valor;
	thread_info = GET_THREAD_INFO (ptask, task, thread);
	WriteFileBuffer_t *wfb = thread_info->file->wfb;

#if !defined(DCARRERA_HADOOP)
	if (!EnabledTasks[ptask - 1][task - 1])
		return;
#endif

	if (type >= MPI_MIN_EV && type <= MPI_MAX_EV)
	{
		Translate_MPI_MPIT2PRV (type, value, &tipus, &valor);
	}
	else
	{
		tipus = type;
		valor = value;
	}
	
	record.type = EVENT;
	record.cpu = cpu;
	record.ptask = ptask;
	record.task = task;
	record.thread = thread_info->virtual_thread;
	record.time = time;
	record.event = tipus;
	record.value = valor;

	trace_paraver_record (wfb, &record);
}

/******************************************************************************
 ***  trace_paraver_unmatched_communication
 ******************************************************************************/
void trace_paraver_unmatched_communication (unsigned int cpu_s, unsigned int ptask_s,
	unsigned int task_s, unsigned int thread_s, unsigned int vthread_s,
	unsigned long long log_s, unsigned long long phy_s, unsigned int cpu_r,
	unsigned int ptask_r, unsigned int task_r, unsigned int thread_r, unsigned int size, unsigned int tag)
{
	thread_t * thread_info_s = GET_THREAD_INFO (ptask_s, task_s, thread_s);
	WriteFileBuffer_t *wfb = thread_info_s->file->wfb;
	paraver_rec_t record;

	UNREFERENCED_PARAMETER(thread_r);

	if (!EnabledTasks[ptask_s-1][task_s-1])
		return;

	record.type = UNMATCHED_COMMUNICATION;
	record.cpu = cpu_s;
	record.ptask = ptask_s;
	record.task = task_s;
	record.thread = vthread_s;
	record.time = log_s;
	record.end_time = phy_s;
	record.event = size;
	record.value = tag;
	record.cpu_r = cpu_r;
	record.ptask_r = ptask_r;
	record.task_r = task_r;
	record.thread_r = thread_r;

	trace_paraver_record (wfb, &record);
}

/******************************************************************************
 ***  trace_paraver_communication
 ******************************************************************************/
void trace_paraver_communication (unsigned int cpu_s, unsigned int ptask_s,
	unsigned int task_s, unsigned int thread_s, unsigned vthread_s, unsigned long long log_s,
	unsigned long long phy_s, unsigned int cpu_r, unsigned int ptask_r,
	unsigned int task_r, unsigned int thread_r, unsigned vthread_r, unsigned long long log_r,
	unsigned long long phy_r, unsigned int size, unsigned int tag,
	int giveOffset, off_t position)
{
	thread_t * thread_info_s = GET_THREAD_INFO (ptask_s, task_s, thread_s);
	WriteFileBuffer_t *wfb = thread_info_s->file->wfb;
	paraver_rec_t record;

	UNREFERENCED_PARAMETER(thread_r);

	if (!(EnabledTasks[ptask_s-1][task_s-1] || EnabledTasks[ptask_r-1][task_r-1]))
		return;

	record.type = COMMUNICATION;
	record.cpu = cpu_s;
	record.ptask = ptask_s;
	record.task = task_s;
	record.thread = vthread_s;
	record.time = log_s;
	record.end_time = phy_s;
	record.event = size;
	record.value = tag;
	record.cpu_r = cpu_r;
	record.ptask_r = ptask_r;
	record.task_r = task_r;
	record.thread_r = vthread_r;
	record.receive[LOGICAL_COMMUNICATION] = log_r;
	record.receive[PHYSICAL_COMMUNICATION] = phy_r;

	if (!giveOffset)
		trace_paraver_record (wfb, &record);
	else
		trace_paraver_recordAt (wfb, &record, position);
}

#if defined(PARALLEL_MERGE)
int trace_paraver_pending_communication (unsigned int cpu_s, 
	unsigned int ptask_s, unsigned int task_s, unsigned int thread_s, unsigned vthread_s,
	unsigned long long log_s, unsigned long long phy_s, unsigned int cpu_r, 
	unsigned int ptask_r, unsigned int task_r, unsigned int thread_r, unsigned vthread_r,
	unsigned long long log_r, unsigned long long phy_r, unsigned int size,
	unsigned int tag)
{
	thread_t * thread_info_s = GET_THREAD_INFO (ptask_s, task_s, thread_s);
	off_t where;
	paraver_rec_t record;
	WriteFileBuffer_t *wfb = thread_info_s->file->wfb;

	UNREFERENCED_PARAMETER(thread_r);
	UNREFERENCED_PARAMETER(log_r);
	UNREFERENCED_PARAMETER(phy_r);

	if (!(EnabledTasks[ptask_s-1][task_s-1] || EnabledTasks[ptask_r-1][task_r-1]))
		return 0;

	record.type = PENDING_COMMUNICATION;
	record.cpu = cpu_s;
	record.ptask = ptask_s;
	record.task = task_s;
	record.thread = vthread_s;
	record.time = log_s;
	record.end_time = phy_s;
	record.event = size;
	record.value = tag;
	record.cpu_r = cpu_r;
	record.ptask_r = ptask_r;
	record.task_r = task_r;
	record.thread_r = vthread_r; /* may need fixing? thread_r instead? */

	/* record.receive[LOGICAL_COMMUNICATION] stores the matching zone (see FixPendingCommunication) */
	record.receive[LOGICAL_COMMUNICATION] = MatchComms_GetZone(ptask_s, task_s);
	record.receive[PHYSICAL_COMMUNICATION] = MatchComms_GetZone(ptask_s, task_s);

	where = WriteFileBuffer_getPosition (wfb);
	AddPendingCommunication (WriteFileBuffer_getFD(wfb), where, tag, task_r-1, task_s-1, MatchComms_GetZone(ptask_s, task_s));
	trace_paraver_record (wfb, &record);

	return 0;
}
#endif

void trace_enter_global_op (unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, unsigned long long time,
	unsigned int com_id, unsigned int send_size, unsigned int recv_size,
	unsigned int is_root)
{
	trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_SENDSIZE, send_size);
	trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_RECVSIZE, recv_size);
	trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_COMM, com_id);
	if (is_root)
		trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_ROOT, is_root);
}

#if defined(NEW_PRINTF)

# include <paraver_nprintf.h>

#endif /* defined(NEW_PRINTF) */


/******************************************************************************
 ***  paraver_state
 ******************************************************************************/
static int paraver_state (struct fdz_fitxer fdz, paraver_rec_t *current)
{
	char buffer[1024];
	int ret;

	unsigned cpu = current->cpu;
	unsigned ptask = current->ptask;
	unsigned task = current->task;
	unsigned thread = current->thread;
	unsigned long long ini_time = current->time;
	unsigned long long end_time = current->end_time;
	unsigned state = current->value;

	CHECK_TIME_US(ini_time);
	CHECK_TIME_US(end_time);

	/*
	 * Format state line is :
	 *      1:cpu:ptask:task:thread:ini_time:end_time:state
	 */
#if !defined(NEW_PRINTF)
# if SIZEOF_LONG == 8
	sprintf (buffer, "1:%d:%d:%d:%d:%lu:%lu:%d\n",
	  cpu, ptask, task, thread, ini_time, end_time, state);
# elif SIZEOF_LONG == 4
	sprintf (buffer, "1:%d:%d:%d:%d:%llu:%llu:%d\n",
	  cpu, ptask, task, thread, ini_time, end_time, state);
# endif
#else /* NEW_PRINTF */
	nprintf_paraver_state (buffer, cpu, ptask, task, thread, ini_time, end_time, state);
#endif /* NEW_PRINTF */

	/* Filter the states with negative or 0 duration */
	if (ini_time < end_time)
	{
		ret = FDZ_WRITE (fdz, buffer);
		if (ret < 0)
		{
			fprintf (stderr, "mpi2prv ERROR : Writing to disk the tracefile\n");
			return -1;
		}
	}
	else if ((int)(end_time - ini_time) < 0)
	{
		fprintf(stderr, "mpi2prv WARNING: Skipping state with negative duration: %s", buffer);
	}
	return 0;
}

/******************************************************************************
 ***  paraver_multi_event
 ******************************************************************************/
static int paraver_multi_event (struct fdz_fitxer fdz, unsigned int cpu,
  unsigned int ptask, unsigned int task, unsigned int thread,
  unsigned long long time, unsigned int count, unsigned int *type,
  UINT64 *value)
{
#if defined(NEW_PRINTF)
	unsigned length;
#endif
  char buffer[1024];
	unsigned i;
	int ret;

  /*
   * Format event line is :
   *      2:cpu:ptask:task:thread:time:(type:value)*
   */

  if (count == 0)
    return 0;

	CHECK_TIME_US(time);

#if !defined(NEW_PRINTF)
# if SIZEOF_LONG == 8
  sprintf (buffer, "2:%d:%d:%d:%d:%lu", cpu, ptask, task, thread, time);
# elif SIZEOF_LONG == 4
  sprintf (buffer, "2:%d:%d:%d:%d:%llu", cpu, ptask, task, thread, time);
# endif
#else /* NEW_PRINTF */
	length = nprintf_paraver_event_head (buffer, cpu, ptask, task, thread, time);
#endif /* NEW_PRINTF */

  ret = FDZ_WRITE (fdz, buffer);
  for (i = 0; i < count; i++)
  {
#if !defined(NEW_PRINTF)
# if SIZEOF_LONG == 8
    sprintf (buffer, ":%d:%lu", type[i], value[i]);
# elif SIZEOF_LONG == 4
    sprintf (buffer, ":%d:%llu", type[i], value[i]);
# endif
#else /* NEW_PRINTF */
		length = nprintf_paraver_event_type_value (buffer, type[i], value[i]);
#endif /* NEW_PRINTF */

    ret = FDZ_WRITE (fdz, buffer);
  }

  ret = FDZ_WRITE (fdz, "\n");

  if (ret < 0)
  {
    fprintf (stderr, "mpi2prv ERROR : Writing to disk the tracefile\n");
    return (-1);
  }
  return 0;
}


/******************************************************************************
 ***  paraver_communication
 ******************************************************************************/
static int paraver_communication (struct fdz_fitxer fdz, paraver_rec_t *current)
{
  char buffer[1024];
  int ret;

	unsigned cpu_s = current->cpu;
	unsigned ptask_s = current->ptask;
	unsigned task_s = current->task;
	unsigned thread_s = current->thread;
	unsigned long long log_s = current->time;
	unsigned long long phy_s = current->end_time;
	unsigned cpu_r = current->cpu_r;
	unsigned ptask_r = current->ptask_r;
	unsigned task_r = current->task_r;
	unsigned thread_r = current->thread_r;
	unsigned long long log_r = current->receive[LOGICAL_COMMUNICATION];
	unsigned long long phy_r = current->receive[PHYSICAL_COMMUNICATION];
	unsigned size = current->event;
	unsigned tag = current->value;

	CHECK_TIME_US(log_s);
	CHECK_TIME_US(phy_s);
	CHECK_TIME_US(log_r);
	CHECK_TIME_US(phy_r);

  /*
   * Format event line is :
   *   3:cpu_s:ptask_s:task_s:thread_s:log_s:phy_s:cpu_r:ptask_r:task_r:
   thread_r:log_r:phy_r:size:tag
   */
#if !defined(NEW_PRINTF)
# if SIZEOF_LONG == 8
  sprintf (buffer, "3:%d:%d:%d:%d:%lu:%lu:%d:%d:%d:%d:%lu:%lu:%d:%d\n",
           cpu_s, ptask_s, task_s, thread_s, log_s, phy_s,
           cpu_r, ptask_r, task_r, thread_r, log_r, phy_r, size, tag);
# elif SIZEOF_LONG == 4
  sprintf (buffer, "3:%d:%d:%d:%d:%llu:%llu:%d:%d:%d:%d:%llu:%llu:%d:%d\n",
           cpu_s, ptask_s, task_s, thread_s, log_s, phy_s,
           cpu_r, ptask_r, task_r, thread_r, log_r, phy_r, size, tag);
# endif
#else /* NEW_PRINTF */

	nprintf_paraver_comm (buffer,
		cpu_s, ptask_s, task_s, thread_s, log_s, phy_s,
		cpu_r, ptask_r, task_r, thread_r, log_r, phy_r,
		size, tag);

#endif /* NEW_PRINTF */

  ret = FDZ_WRITE (fdz, buffer);

  if (ret < 0)
  {
    fprintf (stderr, "mpi2prv ERROR : Writing to disk the tracefile\n");
    return (-1);
  }
  return 0;
}

#if defined(HAVE_BFD)
static UINT64 paraver_translate_bfd_event (unsigned ptask, unsigned task,
	unsigned eventtype, UINT64 eventvalue)
{
	if (eventtype == USRFUNC_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2UF_FUNCTION, get_option_merge_UniqueCallerID());
	else if (eventtype == USRFUNC_LINE_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2UF_LINE, get_option_merge_UniqueCallerID());
	else if (eventtype >= CALLER_EV && eventtype < CALLER_EV + MAX_CALLERS)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2MPI_FUNCTION, get_option_merge_UniqueCallerID());
	else if (eventtype >= CALLER_LINE_EV && eventtype < CALLER_LINE_EV + MAX_CALLERS)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2MPI_LINE, get_option_merge_UniqueCallerID());
	else if (eventtype >= SAMPLING_EV && eventtype < SAMPLING_EV + MAX_CALLERS)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2SAMPLE_FUNCTION, get_option_merge_UniqueCallerID());
	else if (eventtype >= SAMPLING_LINE_EV && eventtype < SAMPLING_LINE_EV + MAX_CALLERS)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2SAMPLE_LINE, get_option_merge_UniqueCallerID());
	else if (eventtype == OMPFUNC_EV || eventtype == TASKFUNC_INST_EV || eventtype == TASKFUNC_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2OMP_FUNCTION, get_option_merge_UniqueCallerID());
	else if (eventtype == OMPFUNC_LINE_EV || eventtype == TASKFUNC_INST_LINE_EV || eventtype == TASKFUNC_LINE_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2OMP_LINE, get_option_merge_UniqueCallerID());
	else if (eventtype == PTHREAD_FUNC_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2OMP_FUNCTION, get_option_merge_UniqueCallerID());
	else if (eventtype == PTHREAD_FUNC_LINE_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2OMP_LINE, get_option_merge_UniqueCallerID());
	else if (eventtype == CUDAFUNC_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2CUDA_FUNCTION, get_option_merge_UniqueCallerID());
	else if (eventtype == CUDAFUNC_LINE_EV)
		return Address2Info_Translate (ptask, task, 
		  eventvalue, ADDR2CUDA_LINE, get_option_merge_UniqueCallerID());
	else
	{
		if (Extrae_Vector_Count (&RegisteredCodeLocationTypes) > 0)
		{
			unsigned u;
			unsigned umax = Extrae_Vector_Count (&RegisteredCodeLocationTypes);
			for (u = 0; u < umax; u++)
			{
				Extrae_Addr2Type_t *element = 
					Extrae_Vector_Get (&RegisteredCodeLocationTypes, u);

				if (element->FunctionType == eventtype)
					return Address2Info_Translate (ptask, task, 
					  eventvalue, element->FunctionType_lbl, get_option_merge_UniqueCallerID());
				else if (element->LineType == eventtype)
					return Address2Info_Translate (ptask, task, 
					  eventvalue, element->LineType_lbl, get_option_merge_UniqueCallerID());
			}
		}
	}

	return eventvalue;
}
#endif /* HAVE_BFD */

static int paraver_build_multi_event (struct fdz_fitxer fdz, paraver_rec_t ** current,
	PRVFileSet_t * fset, unsigned long long *num_events)
{
#define MAX_EVENT_COUNT_IN_MULTI_EVENT	1024
	unsigned int events[MAX_EVENT_COUNT_IN_MULTI_EVENT];
	UINT64 values[MAX_EVENT_COUNT_IN_MULTI_EVENT];
	int prev_cpu, prev_ptask, prev_task, prev_thread;
	unsigned long long prev_time;
	paraver_rec_t *cur;
	UINT64 CallerAddresses[MAX_CALLERS];
	unsigned nevents = 0;

	// Here we store the caller addresses for a reference to a dynamic mem object
	// Set to 0 initially
	memset (CallerAddresses, 0, sizeof(CallerAddresses));

	cur = *current;

	prev_cpu = cur->cpu;
	prev_ptask = cur->ptask;
	prev_task = cur->task;
	prev_thread = cur->thread;
	prev_time = cur->time;

	while (cur != NULL)
	{
	/* Merge multiple events if they are in the same cpu, task, thread and time */
		if (prev_cpu == cur->cpu && prev_ptask == cur->ptask &&
		prev_task == cur->task && prev_thread == cur->thread &&
		prev_time == cur->time && cur->type == 2 &&
		nevents < MAX_EVENT_COUNT_IN_MULTI_EVENT)
		{
			/* Copy the value by default... we'll change it if needed */
			values[nevents] = cur->value;
			events[nevents] = cur->event;

#if defined(DEBUG)
			fprintf (stderr, "mpi2prv: paraver_build_multi_event %d:%d:%d <%d,%llu> @ %llu\n",
			  prev_ptask, prev_task, prev_thread, events[nevents], values[nevents], prev_time);
#endif

			if (cur->event == MPI_GLOBAL_OP_COMM)
				values[nevents] = (UINT64)alies_comunicador ((int) cur->value, cur->ptask, cur->task);
#if defined(HAVE_BFD)
			else
			{
				if (cur->event == USRFUNC_EV || cur->event == USRFUNC_LINE_EV ||
				  (cur->event >= CALLER_EV && cur->event < CALLER_EV + MAX_CALLERS) || 
				  (cur->event >= CALLER_LINE_EV && cur->event < CALLER_LINE_EV + MAX_CALLERS) ||
				  (cur->event >= SAMPLING_EV && cur->event < SAMPLING_EV + MAX_CALLERS) ||
				  (cur->event >= SAMPLING_LINE_EV && cur->event < SAMPLING_LINE_EV + MAX_CALLERS) ||
				  cur->event == OMPFUNC_EV || cur->event == OMPFUNC_LINE_EV ||
				  cur->event == TASKFUNC_EV || cur->event == TASKFUNC_LINE_EV ||
				  cur->event == TASKFUNC_INST_EV || cur->event == TASKFUNC_INST_LINE_EV ||
				  cur->event == PTHREAD_FUNC_EV || cur->event == PTHREAD_FUNC_LINE_EV ||
				  cur->event == CUDAFUNC_EV || cur->event == CUDAFUNC_LINE_EV)
				{
					values[nevents] = paraver_translate_bfd_event (cur->ptask,
					  cur->task, cur->event, cur->value);
				}

				if (cur->event == FILE_NAME_EV)
				{
					/* Unify the file identifiers. Each task stored local identifiers for the open files,
                                         * and after the first merge phase, we shared all the ids and we change them now 
                                         * for a global id, so that each file pathname has an unique id */
					values[nevents] = Unify_File_Id(cur->ptask, cur->task, cur->value);
				}

				if (cur->event >= SAMPLING_ADDRESS_ALLOCATED_OBJECT_CALLER_EV &&
				    cur->event < SAMPLING_ADDRESS_ALLOCATED_OBJECT_CALLER_EV+MAX_CALLERS)
				{
					CallerAddresses[cur->event-SAMPLING_ADDRESS_ALLOCATED_OBJECT_CALLER_EV] =
					  cur->value;
				}

				if (cur->event == SAMPLING_ADDRESS_ALLOCATED_OBJECT_EV)
				{
					values[nevents] = Address2Info_Translate_MemReference (cur->ptask,
					  cur->task, cur->value, MEM_REFERENCE_DYNAMIC,
					  CallerAddresses);

					// Set to 0 again after emitting the information
					memset (CallerAddresses, 0, sizeof(CallerAddresses));
				}
				else if (cur->event == SAMPLING_ADDRESS_STATIC_OBJECT_EV)
				{
					values[nevents] = Address2Info_Translate_MemReference (cur->ptask,
					  cur->task, cur->value, MEM_REFERENCE_STATIC, NULL);
					events[nevents]  = SAMPLING_ADDRESS_ALLOCATED_OBJECT_EV;

					// Set to 0 again after emitting the information
					memset (CallerAddresses, 0, sizeof(CallerAddresses));
				}

				if (Extrae_Vector_Count (&RegisteredCodeLocationTypes) > 0)
				{
					unsigned u;
					unsigned umax = Extrae_Vector_Count (&RegisteredCodeLocationTypes);
					for (u = 0; u < umax; u++)
					{
						Extrae_Addr2Type_t *element = 
							Extrae_Vector_Get (&RegisteredCodeLocationTypes, u);

						if (element->FunctionType == cur->event ||
						    element->LineType == cur->event)
							values[nevents] = paraver_translate_bfd_event (cur->ptask,
							  cur->task, cur->event, cur->value);
					}
				}

				if (get_option_merge_EmitLibraryEvents())
				{
					if (cur->event == USRFUNC_EV ||
					  (cur->event >= CALLER_EV && cur->event < CALLER_EV + MAX_CALLERS) || 
					  (cur->event >= SAMPLING_EV && cur->event < SAMPLING_EV + MAX_CALLERS) ||
					  cur->event == OMPFUNC_EV || cur->event == TASKFUNC_INST_EV ||
					  cur->event == PTHREAD_FUNC_EV || cur->event == CUDAFUNC_EV)
					{
						if (cur->value == UNRESOLVED_ID+1 || cur->value == NOT_FOUND_ID+1)
						{
							nevents++;
							events[nevents] = LIBRARY_EV;
							values[nevents] = Address2Info_GetLibraryID (cur->ptask, cur->task, cur->value);
						}
					}
					else
					{
						if (Extrae_Vector_Count (&RegisteredCodeLocationTypes) > 0)
						{
							unsigned u;
							unsigned umax = Extrae_Vector_Count (&RegisteredCodeLocationTypes);
							for (u = 0; u < umax; u++)
							{
								Extrae_Addr2Type_t *element = 
									Extrae_Vector_Get (&RegisteredCodeLocationTypes, u);
	
								if (element->FunctionType == cur->event || element->LineType == cur->event)
									if (cur->value == UNRESOLVED_ID+1 || cur->value == NOT_FOUND_ID+1)
									{
										nevents++;
										events[nevents] = LIBRARY_EV;
										values[nevents] = Address2Info_GetLibraryID (cur->ptask, cur->task, cur->value);
									}
							}
						}
					}
				}
			}
#endif

			/* These events don't go into final tracefile */
			if (!(cur->event >= SAMPLING_ADDRESS_ALLOCATED_OBJECT_CALLER_EV &&
			    cur->event < SAMPLING_ADDRESS_ALLOCATED_OBJECT_CALLER_EV+MAX_CALLERS))
				nevents++;
		}
		else
			break;

		/* Keep searching ... */
		cur = GetNextParaver_Rec (fset);
	}

	paraver_multi_event (fdz, prev_cpu, prev_ptask, prev_task, prev_thread,
	  prev_time, nevents, events, values);

	*current = cur;
	if (num_events != NULL)
		*num_events = nevents;
	return 0;
#undef MAX_EVENT_COUNT_IN_MULTI_EVENT
}



/******************************************************************************
 *** Paraver_WriteHeader
 ******************************************************************************/
static int Paraver_WriteHeader (FileSet_t *fset, int numtasks, int taskid,
	unsigned num_appl, unsigned long long Ftime, struct fdz_fitxer prv_fd,
	struct Pair_NodeCPU *info)
{
	int NumNodes;
	char Header[1024];
	unsigned threads, task, ptask, node;
	TipusComunicador com;
	int final;

	UNREFERENCED_PARAMETER(numtasks);
#if !defined(PARALLEL_MERGE)
	UNREFERENCED_PARAMETER(fset);
#endif

	if (taskid == 0)
	{
		char Date[80];
		time_t h;

		time (&h);
		strftime (Date, 80, "%d/%m/%Y at %H:%M", localtime (&h));

		/* Write the Paraver header */
#if SIZEOF_LONG == 8
		sprintf (Header, "#Paraver (%s):%lu_ns:", Date, Ftime);
#elif SIZEOF_LONG == 4
		sprintf (Header, "#Paraver (%s):%llu_ns:", Date, Ftime);
#endif
		PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

		NumNodes = 0;
		while (info[NumNodes].files != NULL)
			NumNodes++;

		sprintf (Header, "%d(", NumNodes);
		PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

		if (NumNodes > 0)
		{
			sprintf (Header, "%d", info[0].CPUs);
			PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

			NumNodes = 1;
			while (info[NumNodes].CPUs > 0)
			{
				sprintf (Header,",%d", info[NumNodes].CPUs);
				PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
				NumNodes++;
			}
		}
		sprintf (Header, "):%d:", num_appl);
		PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
	}

	/* For every application, write down its resources */
	for (ptask = 0; ptask < num_appl; ptask++)
	{
#if defined(PARALLEL_MERGE)
		unsigned *vthreads_count = Gather_Paraver_VirtualThreads (taskid, ptask,
			fset);
#endif

		if (taskid == 0)
		{
			ptask_t *ptask_info = GET_PTASK_INFO(ptask+1);
			task_t *last_task_info = GET_TASK_INFO(ptask+1,ptask_info->ntasks);

			sprintf (Header, "%d(", ptask_info->ntasks);
			PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

			for (task = 0; task < ptask_info->ntasks-1; task++)
			{
				task_t *task_info = GET_TASK_INFO(ptask+1,task+1);

#if defined(PARALLEL_MERGE)
				threads = vthreads_count[task];
#else
				threads = task_info->num_virtual_threads; /*nthreads;*/
#endif
				node = task_info->nodeid;

				sprintf (Header, "%d:%d,", threads, node);
				PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
			}
#if defined(PARALLEL_MERGE)
			threads = vthreads_count[ptask_info->ntasks-1];
#else
			threads = last_task_info->num_virtual_threads; /* nthreads */
#endif
			node = last_task_info->nodeid;

			/* Add the communicators info at the last application / ptask */
			if (ptask == num_appl-1)
				sprintf (Header, "%d:%d),%d", threads, node, numero_comunicadors());
			else
				sprintf (Header, "%d:%d),", threads, node);
			PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

#if defined(PARALLEL_MERGE)
			free (vthreads_count);			
#endif
		}
	}

	if (taskid == 0)
	{
		sprintf (Header, "\n");
		PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

		/* Write the communicator definition for every application */
		for (ptask = 1; ptask <= num_appl; ptask++)
		{
			/* Write the communicators created manually by the application */
			final = (primer_comunicador (&com) < 0);
			while (!final)
			{
				unsigned u;

				/* Write this communicator */
				sprintf (Header, "c:%d:%lu:%d", ptask, com.id, com.num_tasks);
				PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
				for (u = 0; u < com.num_tasks; u++)
				{
					sprintf (Header, ":%d", com.tasks[u] + 1);
					PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
				}
				PRVWRITECNTL (FDZ_WRITE (prv_fd, "\n"));

				/* Get the next communicator */
				final = (seguent_comunicador (&com) < 0);
			}

			unsigned pos = 0, hasdata = TRUE;
			while (hasdata)
			{
				uintptr_t intercomm, intracomm1, intracomm2;
				int leader1, leader2;

				hasdata = getInterCommunicatorInfo (pos, &intercomm,
				  &intracomm1, &leader1, &intracomm2, &leader2);

				if (hasdata)
				{
					sprintf (Header, "i:%d:%lu:%lu:%d:%lu:%d\n", ptask, intercomm, intracomm1,
					  leader1, intracomm2, leader2);
					PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
					pos++;
				}
			}
		}
	}

  return 0;
}

#if defined(PARALLEL_MERGE)
static int FixPendingCommunication (paraver_rec_t *current, FileSet_t *fset)
{
	/* Fix parallel pending communication */
	struct ForeignRecv_t* tmp;
	int group;

	group = inWhichGroup (current->ptask_r-1, current->task_r-1, fset);

	/* current->receive[LOGICAL_COMMUNICATION] stores the matching zone (see trace_paraver_pending_communication) */
	tmp = SearchForeignRecv (group, current->ptask-1, current->task-1, current->ptask_r-1, current->task_r-1, current->value, current->receive[LOGICAL_COMMUNICATION]);

	if (NULL != tmp)
	{
		thread_t *thread_r_info = NULL;	

		current->receive[LOGICAL_COMMUNICATION] = tmp->logic;
		current->receive[PHYSICAL_COMMUNICATION] = tmp->physic;
		current->thread_r = tmp->vthread+1; /* AddForeignRecv is called with (vthread-1) */
		thread_r_info = GET_THREAD_INFO(current->ptask_r, current->task_r, current->thread_r);
		current->cpu_r = thread_r_info->cpu; /* The sender cpu is set at trace_paraver_pending_communication */
		current->type = COMMUNICATION;
		tmp->logic = tmp->physic = 0;
		return TRUE;
	}
	return FALSE;
}
#endif

#if defined(DEBUG)
static void DumpUnmatchedCommunication (paraver_rec_t *r)
{
	fprintf (stdout, "UNMATCHED COMUNICATION:\n"
	                 "SENDER: %d.%d RECEIVER: %d.%d TIME/TIMESTAMP %lld/%lld SIZE %d TAG %d\n",
	                 r->task, r->thread, r->task_r, r->thread_r, r->time, r->end_time, r->event, r->value);
}
#endif

static void Paraver_JoinFiles_Master (int numtasks, PRVFileSet_t *prvfset,
	struct fdz_fitxer prv_fd, unsigned long long num_of_events)
{
	/* Master-side. Master will ask all slaves for their parts as needed */
	paraver_rec_t *current;
	double pct, last_pct;
	unsigned long long current_event, tmp;
	int error = FALSE;
	int num_incomplete_state = 0;
	int num_unmatched_comm = 0;
	int num_pending_comm = 0;
	
	fprintf (stdout, "mpi2prv: Generating tracefile (intermediate buffers of %llu events)\n", prvfset->records_per_block);
	fprintf (stdout, "         This process can take a while. Please, be patient.\n");
	if (numtasks > 1)
		fprintf (stdout, "mpi2prv: Progress ... ");
	else
		fprintf (stdout, "mpi2prv: Progress 2 of 2 ... ");
	fflush (stdout);

	current = GetNextParaver_Rec (prvfset);
	current_event = 0;
	last_pct = 0.0f;

	do
	{
		switch (current->type)
		{
			case UNFINISHED_STATE:
			if (num_incomplete_state == 0)
				fprintf (stderr, "mpi2prv: Error! Found an unfinished state in object %d.%d.%d at time %llu (event %llu out of %llu)! Continuing...\n",
				current->ptask, current->task, current->thread, current->time,
				current_event, num_of_events);
			num_incomplete_state++;
			current = GetNextParaver_Rec (prvfset);
			current_event++;
			break;

			case STATE:
			error = paraver_state (prv_fd, current);
			current = GetNextParaver_Rec (prvfset);
			current_event++;
			break;
		
			case EVENT:
			error = paraver_build_multi_event (prv_fd, &current, prvfset, &tmp);
			current_event += tmp;
			break;

			case UNMATCHED_COMMUNICATION:
			if (num_unmatched_comm == 0)
				fprintf (stderr, "mpi2prv: Error! Found unmatched communication! Continuing...\n");
			num_unmatched_comm++;
#if defined(DEBUG)
			DumpUnmatchedCommunication (current);
#endif
			current = GetNextParaver_Rec (prvfset);
			current_event++;
			break;
			
			case PENDING_COMMUNICATION:
#if defined(PARALLEL_MERGE)
			if (FixPendingCommunication (current, prvfset->fset))
				error = paraver_communication (prv_fd, current);
			else
#endif
				num_pending_comm++;

			current = GetNextParaver_Rec (prvfset);
			current_event++;
			break;

			case COMMUNICATION:
			error = paraver_communication (prv_fd, current);
			current = GetNextParaver_Rec (prvfset);
			current_event++;
			break;

			default:
			fprintf(stderr, "\nmpi2prv: Error! Invalid paraver_rec_t (type=%d)\n", current->type);
			exit(-1);
			break;
		}

		pct = ((double) current_event)/((double) num_of_events)*100.0f;

		if (pct > last_pct + 5.0 && pct <= 100.0)
		{
			fprintf (stdout, "%d%% ", (int) pct);
			fflush (stdout);
			while (last_pct + 5.0 < pct)
				last_pct += 5.0;
		}
	}
	while (current != NULL && !error);

	fprintf (stdout, "done\n");
	fflush (stdout);

	if (TimeIn_MicroSecs)
		fprintf (stderr, "mpi2prv: Warning! Clock accuracy seems to be in microseconds instead of nanoseconds.\n");
	if (num_incomplete_state > 0)
		fprintf (stderr, "mpi2prv: Error! Found %d incomplete states. Resulting tracefile may be inconsistent.\n", num_incomplete_state);
	if (num_unmatched_comm > 0)
		fprintf (stderr, "mpi2prv: Error! Found %d unmatched communications. Resulting tracefile may be inconsistent.\n", num_unmatched_comm);
	if (num_pending_comm > 0)
		fprintf (stderr, "mpi2prv: Error! Found %d pending communications. Resulting tracefile may be inconsistent.\n", num_pending_comm);
}

#if defined(PARALLEL_MERGE)
static void Paraver_JoinFiles_Master_Subtree (PRVFileSet_t *prvfset)
{
	/* Master-side. Master will ask all slaves for their parts as needed.
	   This part is ran inside the tree (i.e., non on the root) to generate
	   intermediate binary paraver trace files that the root will transform
	   ASCII regular Paraver files
	 */
	paraver_rec_t *current;

	if (prvfset->SkipAsMasterOfSubtree)
		return;

	current = GetNextParaver_Rec (prvfset);
	do
	{
		if (current->type == PENDING_COMMUNICATION)
		{
			FixPendingCommunication (current, prvfset->fset);
			trace_paraver_record (prvfset->files[0].destination, current);
		}
		else
		{
			trace_paraver_record (prvfset->files[0].destination, current);
		}

		current = GetNextParaver_Rec (prvfset);
	}
	while (current != NULL);
}

static void Paraver_JoinFiles_Slave (PRVFileSet_t *prvfset, int taskid, int tree_fan_out, int current_depth)
{
	/* Slave-side. Master will ask all slaves for their parts as needed */
	paraver_rec_t *current;
	paraver_rec_t *buffer;
	MPI_Status s;
	int res;
	unsigned tmp, nevents;
	int my_master = tree_myMaster (taskid, tree_fan_out, current_depth);

	buffer = malloc (sizeof(paraver_rec_t)*prvfset->records_per_block);
	if (buffer == NULL)
	{
		fprintf (stderr, "mpi2prv: ERROR! Slave %d was unable to allocate %llu bytes to hold records buffer\n", 
			taskid, sizeof(paraver_rec_t)*prvfset->records_per_block);
		fflush (stderr);
		exit (0);
	}

	/* This loop will locally sort the files. Master will only have 
	   to partially sort all the events*/

	nevents = 0;
	current = GetNextParaver_Rec (prvfset);
	do
	{
		if (current->type == PENDING_COMMUNICATION)
			FixPendingCommunication (current, prvfset->fset);

		bcopy (current, &(buffer[nevents++]), sizeof(paraver_rec_t));
		
		if (nevents == prvfset->records_per_block)
		{
			res = MPI_Recv (&tmp, 1, MPI_INT, my_master, ASK_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD, &s);
			MPI_CHECK(res, MPI_Recv, "Failed to receive remote request!");

			res = MPI_Send (&nevents, 1, MPI_UNSIGNED, my_master, HOWMANY_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD); 
			MPI_CHECK(res, MPI_Send, "Failed to send the number of events to the MASTER");

			res = MPI_Send (buffer, nevents*sizeof(paraver_rec_t), MPI_BYTE, my_master, BUFFER_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD); 
			MPI_CHECK(res, MPI_Send, "Failed to send the buffer of events to the MASTER");

			nevents = 0;
		}
		current = GetNextParaver_Rec (prvfset);
	}
	while (current != NULL);
	
	res = MPI_Recv (&tmp, 1, MPI_INT, my_master, ASK_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD, &s);
	MPI_CHECK(res, MPI_Recv, "Failed to receive remote request!");

	res = MPI_Send (&nevents, 1, MPI_UNSIGNED, my_master, HOWMANY_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD); 
	MPI_CHECK(res, MPI_Send, "Failed to send the number of events to the MASTER");

	if (nevents != 0)
	{
		res = MPI_Send (buffer, nevents*sizeof(paraver_rec_t), MPI_BYTE, my_master, BUFFER_MERGE_REMOTE_BLOCK_TAG, MPI_COMM_WORLD); 
		MPI_CHECK(res, MPI_Send, "Failed to send the buffer of events to the MASTER");
	}

	free (buffer);
}
#endif

/******************************************************************************
 ***  Paraver_JoinFiles
 ******************************************************************************/

int Paraver_JoinFiles (unsigned num_appl, char *outName, FileSet_t * fset,
	unsigned long long Ftime, struct Pair_NodeCPU *NodeCPUinfo, int numtasks,
	int taskid, unsigned long long records_per_task, int tree_fan_out)
{
	time_t delta;
	struct timeval time_begin, time_end;
#if defined(PARALLEL_MERGE)
	int res;
	int tree_max_depth;
	int current_depth;
#endif
	PRVFileSet_t *prvfset = NULL;
	unsigned long long num_of_events;
	struct fdz_fitxer prv_fd;
	int error = FALSE;
#if defined(IS_BG_MACHINE)
	FILE *crd_fd;
	int i;
	char envName[PATH_MAX], *tmpName;
#endif

#if !defined(PARALLEL_MERGE)
	UNREFERENCED_PARAMETER(tree_fan_out);
#endif

	prv_fd.handle = NULL;
#ifdef HAVE_ZLIB
	prv_fd.handleGZ = NULL;
#endif

	if (0 == taskid)
	{
#ifdef HAVE_ZLIB
		if (strlen (outName) >= 7 &&
		strncmp (&(outName[strlen (outName) - 7]), ".prv.gz", 7) == 0)
		{
			/*
			* Open GZ handle for the file, and mark normal handle as unused! 
			*/
			prv_fd.handleGZ = gzopen (outName, "wb6");
			prv_fd.handle = NULL;
			if (prv_fd.handleGZ == NULL)
			{
				fprintf (stderr, "mpi2prv ERROR: creating GZ paraver tracefile : %s\n",
					outName);
				exit (-1);
			}
		}
		else
		{
			/*
			* Open normal handle for the file, and mark GZ handle as unused! 
			*/
#if HAVE_FOPEN64
			prv_fd.handle = fopen64 (outName, "w");
#else
			prv_fd.handle = fopen (outName, "w");
#endif
			prv_fd.handleGZ = NULL;
			if (prv_fd.handle == NULL)
			{
				fprintf (stderr, "mpi2prv ERROR: Creating Paraver tracefile : %s\n",
					outName);
				exit (-1);
			}
		}
#else

		/* If the user requested .prv.gz but it is not supported, change into .prv */
		if (strlen (outName) >= 7 &&
		strncmp (&(outName[strlen (outName) - 7]), ".prv.gz", 7) == 0)
			outName[strlen(outName)-3] = (char) 0;

		/* Open normal handle for the file, and mark GZ handle as unused!  */
#if HAVE_FOPEN64
		prv_fd.handle = fopen64 (outName, "w");
#else
		prv_fd.handle = fopen (outName, "w");
#endif
		if (prv_fd.handle == NULL)
		{
			fprintf (stderr, "mpi2prv ERROR: Creating Paraver tracefile : %s\n",
				outName);
			exit (-1);
		}
#endif
	} /* taskid == 0 */

	error = Paraver_WriteHeader (fset, numtasks, taskid, num_appl, Ftime, prv_fd,
	  NodeCPUinfo);

#if defined(PARALLEL_MERGE)
	res = MPI_Bcast (&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Error! Failed to propagate status");
#endif
	if (error)
		return -1;

#if defined(PARALLEL_MERGE)
	tree_max_depth = tree_MaxDepth (numtasks, tree_fan_out);
	if (taskid == 0)
		fprintf (stdout, "mpi2prv: Merge tree depth for %d tasks is %d levels using a fan-out of %d leaves\n", numtasks, tree_max_depth, tree_fan_out);

	current_depth = 0;
	while (current_depth < tree_max_depth)
	{
		if (taskid == 0)
		{
			gettimeofday (&time_begin, NULL);
			fprintf (stdout, "mpi2prv: Executing merge tree step %d of %d.\n", current_depth+1, tree_max_depth);
		}

		if (tree_TaskHaveWork (taskid, tree_fan_out, current_depth))
		{
			if (current_depth == 0)
				prvfset = Map_Paraver_files (fset, &num_of_events, numtasks, taskid, records_per_task, tree_fan_out);
			else
				prvfset = ReMap_Paraver_files_binary (prvfset, &num_of_events, numtasks, taskid, records_per_task, current_depth, tree_fan_out);

			if (!tree_MasterOfSubtree (taskid, tree_fan_out, current_depth))
			{
				/* Server-side. Slaves will merge their translated files into a 
				   single strem and will provide it to the master */
				Paraver_JoinFiles_Slave (prvfset, taskid, tree_fan_out, current_depth);
			}
			else
			{
				/* If this is not the root level, only generate binary intermediate files */
				if (current_depth < tree_max_depth-1)
					Paraver_JoinFiles_Master_Subtree (prvfset);
				else
					Paraver_JoinFiles_Master (numtasks, prvfset, prv_fd, num_of_events);
			}

			Free_Map_Paraver_Files (prvfset);
		}
		else
		{
			/* Do nothing */
		}

		res = MPI_Barrier (MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Barrier, "Failed to step to the next tree level")
		if (taskid == 0)
		{
			gettimeofday (&time_end, NULL);

			delta = time_end.tv_sec - time_begin.tv_sec;
			fprintf (stdout, "mpi2prv: Elapsed time on tree step %d: %ld hours %ld minutes %ld seconds\n", current_depth+1, delta / 3600, (delta % 3600)/60, (delta % 60));
		}

		Flush_Paraver_Files_binary (prvfset, taskid, current_depth, tree_fan_out);
		current_depth++;
	}

#else /* PARALLEL_MERGE */

	gettimeofday (&time_begin, NULL);

	prvfset = Map_Paraver_files (fset, &num_of_events, numtasks, taskid, records_per_task);

	Paraver_JoinFiles_Master (numtasks, prvfset, prv_fd, num_of_events);

	gettimeofday (&time_end, NULL);
	delta = time_end.tv_sec - time_begin.tv_sec;
	fprintf (stdout, "mpi2prv: Elapsed time merge step: %ld hours %ld minutes %ld seconds\n", delta / 3600, (delta % 3600)/60, (delta % 60));

#endif /* PARALLEL_MERGE */

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Resulting tracefile occupies %lld bytes\n", (long long) FDZ_TELL(prv_fd));
		FDZ_CLOSE (prv_fd);
	}

	Free_FS (fset);

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Removing temporal files... ");
		fflush (stdout);
		gettimeofday (&time_begin, NULL);
	}
	WriteFileBuffer_deleteall();
#if defined(PARALLEL_MERGE)
	res = MPI_Barrier (MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Barrier, "Failed to step to the next tree level")
#endif
	if (taskid == 0)
	{
		gettimeofday (&time_end, NULL);
		fprintf (stdout, "done\n");
		fflush (stdout);
		delta = time_end.tv_sec - time_begin.tv_sec;
		fprintf (stdout, "mpi2prv: Elapsed time removing temporal files: %ld hours %ld minutes %ld seconds\n", delta / 3600, (delta % 3600)/60, (delta % 60));
	}

#if defined(IS_BG_MACHINE)
#if defined(DEAD_CODE)
	/* FIXME must be implemented in parallel */
	if (get_option_merge_XYZT())
	{
		strcpy (envName, get_OutputTraceName());

		if (strlen (outName) >= 7 &&
			strncmp (&(outName[strlen (outName) - 7]), ".prv.gz", 7) == 0)
			tmpName = &(envName[strlen (envName) - 7]);
		else
			tmpName = &(envName[strlen (envName) - 4]);

		strcpy (tmpName, ".crd");
		if ((crd_fd = fopen (envName, "w")) == NULL)
		{
			fprintf (stderr, "mpi2prv ERROR: Creating coordinates file : %s\n", tmp);
			return 0;
		}

		for (i = 0; i < nfiles; i++)
			fprintf (crd_fd, "%d %d %d %d\n", coords[i].X, coords[i].Y, coords[i].Z,
			coords[i].T);
		fclose (crd_fd);
		free (coords);
		fprintf (stdout, "\nCoordinate file generated\n");
	}
#endif /* FIXME */
#endif

	if (0 == taskid)
		if (error)
			unlink (outName);
	return 0;
}
