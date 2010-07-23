/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

#include "mpi_prv_events.h"
#include "pacx_prv_events.h"
#include "addr2info.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
# include "tree-logistics.h"
#endif

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
 ***  trace_paraver_state
 ******************************************************************************/
void trace_paraver_state (
   unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
   unsigned long long current_time)
{
	struct thread_t * thread_info;
	unsigned int current_state;
	WriteFileBuffer_t *wfb = obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb;

	thread_info = GET_THREAD_INFO (ptask, task, thread);
	current_state = Top_State(ptask, task, thread);

	/* Complete the previous state */
	if (thread_info->incomplete_state_offset != (off_t)-1) /* This isn't the first state */
	{
		/* Do not split states whether appropriate */
		if (Get_Joint_States() && !Get_Last_State())
			if (thread_info->incomplete_state_record.value == current_state)
				return;
		/* Write the record into the *.tmp file if the state isn't excluded */
#if defined(DEBUG_STATES)
		fprintf(stderr, "mpi2prv: DEBUG [T:%d] Closing state %u at %llu\n", task,  
		(unsigned int)thread_info->incomplete_state_record.value, current_time);
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
	thread_info->incomplete_state_record.thread = thread;
	thread_info->incomplete_state_record.time   = current_time;
	thread_info->incomplete_state_record.value  = current_state;
	/* Save a slot in the *.tmp file for this record if this state isn't excluded */
#if defined(DEBUG_STATES)
	fprintf(stderr, "mpi2prv: DEBUG [T:%d] Starting state %u at %llu\n", task, current_state, current_time);
#endif
	if (!State_Excluded(current_state))
	{
		paraver_rec_t fake_record;
		fake_record.type = UNFINISHED_STATE;
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
	paraver_rec_t record;
	int tipus;
	UINT64 valor;
	WriteFileBuffer_t *wfb = obj_table[ptask-1].tasks[task-1].threads[thread-1].file->wfb;

	if (!EnabledTasks[ptask - 1][task - 1])
		return;

	if (type >= MPI_MIN_EV && type <= MPI_MAX_EV)
	{
		Translate_MPI_MPIT2PRV (type, value, &tipus, &valor);
	}
	else if (type >= PACX_MIN_EV && type <= PACX_MAX_EV)
	{
		Translate_PACX_MPIT2PRV (type, value, &tipus, &valor);
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
	record.thread = thread;
	record.time = time;
	record.event = tipus;
	record.value = valor;

	trace_paraver_record (wfb, &record);
}

/******************************************************************************
 ***  trace_paraver_unmatched_communication
 ******************************************************************************/
void trace_paraver_unmatched_communication (unsigned int cpu_s, unsigned int ptask_s,
	unsigned int task_s, unsigned int thread_s, unsigned long long log_s,
	unsigned long long phy_s, unsigned int cpu_r, unsigned int ptask_r,
	unsigned int task_r, unsigned int thread_r, unsigned int size, unsigned int tag)
{
	WriteFileBuffer_t *wfb = obj_table[ptask_s-1].tasks[task_s-1].threads[thread_s-1].file->wfb;
	paraver_rec_t record;

	if (!EnabledTasks[ptask_s-1][task_s-1])
		return;

	record.type = UNMATCHED_COMMUNICATION;
	record.cpu = cpu_s;
	record.ptask = ptask_s;
	record.task = task_s;
	record.thread = thread_s;
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
	unsigned int task_s, unsigned int thread_s, unsigned long long log_s,
	unsigned long long phy_s, unsigned int cpu_r, unsigned int ptask_r,
	unsigned int task_r, unsigned int thread_r, unsigned long long log_r,
	unsigned long long phy_r, unsigned int size, unsigned int tag,
	int giveOffset, off_t position)
{
	WriteFileBuffer_t *wfb = obj_table[ptask_s-1].tasks[task_s-1].threads[thread_s-1].file->wfb;
	paraver_rec_t record;

	if (!(EnabledTasks[ptask_s-1][task_s-1] || EnabledTasks[ptask_r-1][task_r-1]))
		return;

	record.type = COMMUNICATION;
	record.cpu = cpu_s;
	record.ptask = ptask_s;
	record.task = task_s;
	record.thread = thread_s;
	record.time = log_s;
	record.end_time = phy_s;
	record.event = size;
	record.value = tag;
	record.cpu_r = cpu_r;
	record.ptask_r = ptask_r;
	record.task_r = task_r;
	record.thread_r = thread_r;
	record.receive[LOGICAL_COMMUNICATION] = log_r;
	record.receive[PHYSICAL_COMMUNICATION] = phy_r;

	if (!giveOffset)
		trace_paraver_record (wfb, &record);
	else
		trace_paraver_recordAt (wfb, &record, position);
}

#if defined(PARALLEL_MERGE)
int trace_paraver_pending_communication (unsigned int cpu_s, 
	unsigned int ptask_s, unsigned int task_s, unsigned int thread_s,
	unsigned long long log_s, unsigned long long phy_s, unsigned int cpu_r, 
	unsigned int ptask_r, unsigned int task_r, unsigned int thread_r,
	unsigned long long log_r, unsigned long long phy_r, unsigned int size,
	unsigned int tag)
{
	off_t where;
	paraver_rec_t record;
	WriteFileBuffer_t *wfb = obj_table[ptask_s-1].tasks[task_s-1].threads[thread_s-1].file->wfb;

	UNREFERENCED_PARAMETER(log_r);
	UNREFERENCED_PARAMETER(phy_r);

	if (!(EnabledTasks[ptask_s-1][task_s-1] || EnabledTasks[ptask_r-1][task_r-1]))
		return 0;

	record.type = PENDING_COMMUNICATION;
	record.cpu = cpu_s;
	record.ptask = ptask_s;
	record.task = task_s;
	record.thread = thread_s;
	record.time = log_s;
	record.end_time = phy_s;
	record.event = size;
	record.value = tag;
	record.cpu_r = cpu_r;
	record.ptask_r = ptask_r;
	record.task_r = task_r;
	record.thread_r = thread_r;
  record.receive[LOGICAL_COMMUNICATION] = 0;
  record.receive[PHYSICAL_COMMUNICATION] = 0;

	where = WriteFileBuffer_getPosition (wfb);
	AddPendingCommunication (WriteFileBuffer_getFD(wfb), where, tag, task_r-1, task_s-1);
	trace_paraver_record (wfb, &record);

	return 0;
}
#endif

void trace_enter_global_op (unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, unsigned long long time,
	unsigned int com_id, unsigned int send_size, unsigned int recv_size,
	unsigned int is_root, unsigned isMPI)
{
	if (isMPI)
	{
		trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_SENDSIZE, send_size);
		trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_RECVSIZE, recv_size);
		trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_COMM, com_id);
		if (is_root)
			trace_paraver_event (cpu, ptask, task, thread, time, MPI_GLOBAL_OP_ROOT, is_root);
	}
	else
	{
		trace_paraver_event (cpu, ptask, task, thread, time, PACX_GLOBAL_OP_SENDSIZE, send_size);
		trace_paraver_event (cpu, ptask, task, thread, time, PACX_GLOBAL_OP_RECVSIZE, recv_size);
		trace_paraver_event (cpu, ptask, task, thread, time, PACX_GLOBAL_OP_COMM, com_id);
		if (is_root)
			trace_paraver_event (cpu, ptask, task, thread, time, PACX_GLOBAL_OP_ROOT, is_root);
	}
}

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

	/*
	 * Format state line is :
	 *      1:cpu:ptask:task:thread:ini_time:end_time:state
	 */
#if SIZEOF_LONG == 8
	sprintf (buffer, "1:%d:%d:%d:%d:%lu:%lu:%d\n",
	  cpu, ptask, task, thread, ini_time, end_time, state);
#elif SIZEOF_LONG == 4
	sprintf (buffer, "1:%d:%d:%d:%d:%llu:%llu:%d\n",
	  cpu, ptask, task, thread, ini_time, end_time, state);
#endif

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


#if defined(DEAD_CODE)
/******************************************************************************
 ***  paraver_event
 ******************************************************************************/
int paraver_event (struct fdz_fitxer fdz, unsigned int cpu,
                   unsigned int ptask, unsigned int task, unsigned int thread,
                   unsigned long long time, unsigned int type,
                   UINT64 value)
{
  char buffer[1024];
  int ret;

  /*
   * Format event line is :
   *      2:cpu:ptask:task:thread:time:type:value
   */
#if SIZEOF_LONG == 8
  sprintf (buffer, "2:%d:%d:%d:%d:%lu:%d:%lu\n",
           cpu, ptask, task, thread, time, type, value);
#elif SIZEOF_LONG == 4
  sprintf (buffer, "2:%d:%d:%d:%d:%llu:%d:%llu\n",
           cpu, ptask, task, thread, time, type, value);
#endif

  ret = FDZ_WRITE (fdz, buffer);

  if (ret < 0)
  {
    fprintf (stderr, "mpi2prv ERROR: Writing to disk the tracefile\n");
    return -1;
  }
  return 0;
}
#endif /* DEAD_CODE */

/******************************************************************************
 ***  paraver_multi_event
 ******************************************************************************/
static int paraver_multi_event (struct fdz_fitxer fdz, unsigned int cpu,
  unsigned int ptask, unsigned int task, unsigned int thread,
  unsigned long long time, unsigned int count, unsigned int *type,
  UINT64 *value)
{
  char buffer[1024];
  int i, ret;

  /*
   * Format event line is :
   *      2:cpu:ptask:task:thread:time:(type:value)*
   */

  if (count == 0)
    return 0;

#if SIZEOF_LONG == 8
  sprintf (buffer, "2:%d:%d:%d:%d:%lu", cpu, ptask, task, thread, time);
#elif SIZEOF_LONG == 4
  sprintf (buffer, "2:%d:%d:%d:%d:%llu", cpu, ptask, task, thread, time);
#endif
  ret = FDZ_WRITE (fdz, buffer);
  for (i = 0; i < count; i++)
  {
#if SIZEOF_LONG == 8
    sprintf (buffer, ":%d:%lu", type[i], value[i]);
#elif SIZEOF_LONG == 4
    sprintf (buffer, ":%d:%llu", type[i], value[i]);
#endif
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

  /*
   * Format event line is :
   *   3:cpu_s:ptask_s:task_s:thread_s:log_s:phy_s:cpu_r:ptask_r:task_r:
   thread_r:log_r:phy_r:size:tag
   */
#if SIZEOF_LONG == 8
  sprintf (buffer, "3:%d:%d:%d:%d:%lu:%lu:%d:%d:%d:%d:%lu:%lu:%d:%d\n",
           cpu_s, ptask_s, task_s, thread_s, log_s, phy_s,
           cpu_r, ptask_r, task_r, thread_r, log_r, phy_r, size, tag);
#elif SIZEOF_LONG == 4
  sprintf (buffer, "3:%d:%d:%d:%d:%llu:%llu:%d:%d:%d:%d:%llu:%llu:%d:%d\n",
           cpu_s, ptask_s, task_s, thread_s, log_s, phy_s,
           cpu_r, ptask_r, task_r, thread_r, log_r, phy_r, size, tag);
#endif

  ret = FDZ_WRITE (fdz, buffer);

  if (ret < 0)
  {
    fprintf (stderr, "mpi2prv ERROR : Writing to disk the tracefile\n");
    return (-1);
  }
  return 0;
}

#if defined(HAVE_BFD)
static UINT64 translate_bfd_event (unsigned eventtype, UINT64 eventvalue)
{
	if (eventtype == USRFUNC_EV)
		return Address2Info_Translate (eventvalue, ADDR2UF_FUNCTION, option_UniqueCallerID);
	else if (eventtype >= CALLER_EV && eventtype < CALLER_EV + MAX_CALLERS)
		return Address2Info_Translate (eventvalue, ADDR2MPI_FUNCTION, option_UniqueCallerID);
	else if (eventtype >= CALLER_LINE_EV && eventtype < CALLER_LINE_EV + MAX_CALLERS)
		return Address2Info_Translate (eventvalue, ADDR2MPI_LINE, option_UniqueCallerID);
	else if (eventtype == USRFUNC_LINE_EV)
		return Address2Info_Translate (eventvalue, ADDR2UF_LINE, option_UniqueCallerID);
	else if (eventtype >= SAMPLING_EV && eventtype < SAMPLING_EV + MAX_CALLERS)
		return Address2Info_Translate (eventvalue, ADDR2SAMPLE_FUNCTION, option_UniqueCallerID);
	else if (eventtype >= SAMPLING_LINE_EV && eventtype < SAMPLING_LINE_EV + MAX_CALLERS)
		return Address2Info_Translate (eventvalue, ADDR2SAMPLE_LINE, option_UniqueCallerID);
	else if (eventtype == OMPFUNC_EV)
		return Address2Info_Translate (eventvalue, ADDR2OMP_FUNCTION, option_UniqueCallerID);
	else if (eventtype == OMPFUNC_LINE_EV)
		return Address2Info_Translate (eventvalue, ADDR2OMP_LINE, option_UniqueCallerID);
	else if (eventtype == PTHREADFUNC_EV)
		return Address2Info_Translate (eventvalue, ADDR2OMP_FUNCTION, option_UniqueCallerID);
	else if (eventtype == PTHREADFUNC_LINE_EV)
		return Address2Info_Translate (eventvalue, ADDR2OMP_LINE, option_UniqueCallerID);
	else
		return eventvalue;
}
#endif /* HAVE_BFD */

static int build_multi_event (struct fdz_fitxer fdz, paraver_rec_t ** current,
	PRVFileSet_t * fset, unsigned long long *num_events)
{
#define MAX_EVENT_COUNT_IN_MULTI_EVENT	1024
	unsigned int events[MAX_EVENT_COUNT_IN_MULTI_EVENT];
	UINT64 values[MAX_EVENT_COUNT_IN_MULTI_EVENT];
	int prev_cpu, prev_ptask, prev_task, prev_thread;
	unsigned long long prev_time;
	paraver_rec_t *cur;
	int i = 0;

	cur = *current;

	prev_cpu = cur->cpu;
	prev_ptask = cur->ptask;
	prev_task = cur->task;
	prev_thread = cur->thread;
	prev_time = cur->time;

	while (cur != NULL)
	{
#if defined(IS_BG_MACHINE)
	/*
	* Per cadascun dels events, hem de comprovar si cal anotar les
	* localitzacions del torus a la trasa 
	*/
		if (cur->type == 2 && option_XYZT
		&& (cur->event == BG_PERSONALITY_TORUS_X
		|| cur->event == BG_PERSONALITY_TORUS_Y
		|| cur->event == BG_PERSONALITY_TORUS_Z
		|| cur->event == BG_PERSONALITY_PROCESSOR_ID))
		AnotaBGPersonality (cur->event, cur->value, cur->task);
#endif


	/* Merge multiple events if they are in the same cpu, task, thread and time */
		if (prev_cpu == cur->cpu && prev_ptask == cur->ptask &&
		prev_task == cur->task && prev_thread == cur->thread &&
		prev_time == cur->time && cur->type == 2 &&
		i < MAX_EVENT_COUNT_IN_MULTI_EVENT)
		{
			/* Copy the value by default... we'll change it if needed */
			values[i] = cur->value;
			events[i] = cur->event;

			if (cur->event == MPI_GLOBAL_OP_COMM)
				values[i] = (UINT64)alies_comunicador ((int) cur->value, cur->ptask, cur->task);
#if defined(HAVE_BFD)
			else
				values[i] = translate_bfd_event (cur->event, cur->value);
#endif
			i++;
		}
		else
			break;

		/* Keep searching ... */
		cur = GetNextParaver_Rec (fset);
	}

	paraver_multi_event (fdz, prev_cpu, prev_ptask, prev_task, prev_thread,
	  prev_time, i, events, values);

	*current = cur;
	if (num_events != NULL)
		*num_events = i;
	return 0;
#undef MAX_EVENT_COUNT_IN_MULTI_EVENT
}



/******************************************************************************
 *** Paraver_WriteHeader
 ******************************************************************************/
static int Paraver_WriteHeader (unsigned long long Ftime,
  struct fdz_fitxer prv_fd, struct Pair_NodeCPU *info)
{
	int NumNodes;
  time_t h;
  char Date[80];
  char Header[1024];
  unsigned int threads, task, ptask, node, num_cpus = 1;
#if defined(HAVE_MPI)  /* Sequential tracing does not use comunicators */
  TipusComunicador com;
  int i, final;
  unsigned int num_tasks;
#endif

  time (&h);
  strftime (Date, 80, "%d/%m/%Y at %H:%M", localtime (&h));

  for (ptask = 0; ptask < num_ptasks; ptask++)
    num_cpus = MAX (num_cpus, obj_table[ptask].ntasks);

  /* Write the Paraver header */
#if SIZEOF_LONG == 8
  sprintf (Header, "#Paraver (%s):%lu_ns:", Date, Ftime);
#elif SIZEOF_LONG == 4
  sprintf (Header, "#Paraver (%s):%llu_ns:", Date, Ftime);
#endif
	PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

	NumNodes = 0;
	while (info[NumNodes].NodeName != NULL)
		NumNodes++;

	sprintf (Header, "%d(", NumNodes);
	PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

	if (NumNodes > 0)
	{
		sprintf (Header, "%d", info[0].CPUs);
		PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

		NumNodes = 1;
		while (info[NumNodes].NodeName != NULL)
		{
			sprintf (Header,",%d", info[NumNodes].CPUs);
			PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
			NumNodes++;
		}
	}
	sprintf (Header, "):%d:", num_ptasks);
	PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

	/* For every application, write down its resources */
  for (ptask = 0; ptask < num_ptasks; ptask++)
  {
    sprintf (Header, "%d(", obj_table[ptask].ntasks);
		PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));

    for (task = 0; task < obj_table[ptask].ntasks - 1; task++)
    {
      threads = obj_table[ptask].tasks[task].nthreads;
      node = obj_table[ptask].tasks[task].nodeid;

      sprintf (Header, "%d:%d,", threads, node);
			PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
    }
    threads = obj_table[ptask].tasks[obj_table[ptask].ntasks-1].nthreads;
    node =  obj_table[ptask].tasks[obj_table[ptask].ntasks-1].nodeid;

#if defined(HAVE_MPI)
    sprintf (Header, "%d:%d),%d", threads, node, numero_comunicadors());
#else
    sprintf (Header, "%d:%d),0", threads, node);
#endif
		PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
  }
  sprintf (Header, "\n");
  PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));


#if defined(HAVE_MPI)
	/* Write the communicator definition for every application */
  for (ptask = 1; ptask <= num_ptasks; ptask++)
  {
    num_tasks = obj_table[ptask - 1].ntasks;

		/* Write the communicators created manually by the application */
    final = (primer_comunicador (&com) < 0);
    while (!final)
    {
      /* Write this communicator */
      sprintf (Header, "c:%d:%d:%d", ptask, com.id, com.num_tasks);
      PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
      for (i = 0; i < com.num_tasks; i++)
      {
        sprintf (Header, ":%d", com.tasks[i] + 1);
        PRVWRITECNTL (FDZ_WRITE (prv_fd, Header));
      }
      PRVWRITECNTL (FDZ_WRITE (prv_fd, "\n"));

      /* Get the next communicator */
      final = (seguent_comunicador (&com) < 0);
    }
  }
#endif

  return 0;
}

#if defined(PARALLEL_MERGE)
static int FixPendingCommunication (paraver_rec_t *current, FileSet_t *fset)
{
	/* Fix parallel pending communication */
	struct ForeignRecv_t* tmp;
	int group;

	group = inWhichGroup (current->task_r-1, fset);

	tmp = SearchForeignRecv (group, current->task-1, current->task_r-1, current->value);

	if (NULL != tmp)
	{
		current->receive[LOGICAL_COMMUNICATION] = tmp->logic;
		current->receive[PHYSICAL_COMMUNICATION] = tmp->physic;
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
	int error;
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
				fprintf (stderr, "mpi2prv: Error! Found an unfinished state! Continuing...\n");
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
			error = build_multi_event (prv_fd, &current, prvfset, &tmp);
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
			fprintf (stdout, "%.1lf%% ", pct);
			fflush (stdout);
			last_pct += 5.0;
		}
	}
	while (current != NULL && !error);

	fprintf (stdout, "done\n");
	fflush (stdout);

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

int Paraver_JoinFiles (char *outName, FileSet_t * fset, unsigned long long Ftime,
  struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int taskid,
  unsigned long long records_per_task, int tree_fan_out)
{
#if defined(PARALLEL_MERGE)
	struct timeval time_begin, time_end;
	int res;
	int tree_max_depth;
	int current_depth;
#endif
	PRVFileSet_t *prvfset;
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
				return -1;
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
				return -1;
			}
		}
#else
		/*
		* Open normal handle for the file, and mark GZ handle as unused! 
		*/
#if HAVE_FOPEN64
		prv_fd.handle = fopen64 (outName, "w");
#else
		prv_fd.handle = fopen (outName, "w");
#endif
		if (prv_fd.handle == NULL)
		{
			fprintf (stderr, "mpi2prv ERROR: Creating Paraver tracefile : %s\n",
				outName);
			return -1;
		}
#endif
	} /* taskid == 0 */

#if defined(IS_BG_MACHINE)
#if defined(DEAD_CODE)
	/* FIXME must be implemented in parallel */
	if (option_XYZT)
	{
		coords = (struct QuadCoord *) malloc (nfiles * sizeof (struct QuadCoord));
		if (coords == NULL)
		{
			fprintf (stderr,
			"mpi2prv: ERROR: Unable to allocate memory for coordinates file\n");
			return -1;
		}
	}
#endif /* FIXME */
#endif

	if (0 == taskid)
		error = Paraver_WriteHeader (Ftime, prv_fd, NodeCPUinfo);

#if defined(PARALLEL_MERGE)
	res = MPI_Bcast (&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Error! Failed to propagate status");
#endif
	if (error)
		return -1;

#if defined(PARALLEL_MERGE)
	tree_max_depth = tree_MaxDepth (numtasks, tree_fan_out);
	if (taskid == 0)
		fprintf (stdout, "mpi2prv: Merge tree depth is %d levels\n", tree_max_depth);

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

		MPI_Barrier (MPI_COMM_WORLD);
		if (taskid == 0)
		{
			time_t delta;
			gettimeofday (&time_end, NULL);

			delta = time_end.tv_sec - time_begin.tv_sec;
			fprintf (stdout, "mpi2prv: Elapsed time on tree step %d: %d hours %d minutes %d seconds\n", current_depth+1, delta / 3600, (delta % 3600)/60, (delta % 60));
		}

		Flush_Paraver_Files_binary (prvfset, taskid, current_depth, tree_fan_out);
		current_depth++;
	}

#else /* PARALLEL_MERGE */

	prvfset = Map_Paraver_files (fset, &num_of_events, numtasks, taskid, records_per_task);

	Paraver_JoinFiles_Master (numtasks, prvfset, prv_fd, num_of_events);

#endif /* PARALLEL_MERGE */

	if (taskid == 0)
		FDZ_CLOSE (prv_fd);

	Free_FS (fset);

#if defined(IS_BG_MACHINE)
#if defined(DEAD_CODE)
	/* FIXME must be implemented in parallel */
	if (option_XYZT)
	{
		strcpy (envName, outName);

#ifdef HAVE_ZLIB
		if (strlen (outName) >= 7 &&
			strncmp (&(outName[strlen (outName) - 7]), ".prv.gz", 7) == 0)
			tmpName = &(envName[strlen (envName) - 7]);
		else
			tmpName = &(envName[strlen (envName) - 4]);
#else
		tmpName = &(envName[strlen (envName) - 4]);
#endif

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
