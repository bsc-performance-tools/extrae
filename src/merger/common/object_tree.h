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

#ifndef _OBJECT_TREE_H
#define _OBJECT_TREE_H

#include "common.h"

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "file_set.h"
#include "new-queue.h"
#include "HardwareCounters.h"
#include "communication_queues.h"

#define MAX_STATES 16

#define GET_TASK_INFO(ptask, task) \
    &(obj_table[ptask - 1].tasks[task - 1])

#define GET_THREAD_INFO(ptask, task, thread) \
    &(obj_table[ptask - 1].tasks[task - 1].threads[thread - 1])

typedef struct thread_t
{
	/* Where is this thread running? */
	unsigned int cpu;

	/* Did we passed the first event? */
	unsigned int First_Event:1;

	/* Did we processed the firt HWC change event? */
	unsigned int First_HWCChange:1;

	/* Information of the stack */
	int State_Stack[MAX_STATES];
	int nStates;

	/* Controls whether this thread matches comms or not */
	int MatchingComms;

	/* Stores the time of the last event */
	UINT64 Previous_Event_Time;

	/* Information of the last state record */
	paraver_rec_t incomplete_state_record;
	off_t incomplete_state_offset;

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	unsigned long long last_hw_group_change;
	int **HWCSets;
	int num_HWCSets;
	int current_HWCSet;
	long long counters[MAX_HWC];  /* HWC values */
#endif

	event_t *Send_Rec;  /* Store send records */
	event_t *Recv_Rec;  /* Store receive records */
	FileItem_t *file;

	unsigned long long dimemas_size; /* Store dimemas translation size */

	unsigned virtual_thread; /* if so, which virtual thread is? */
} thread_t;

typedef struct task_t
{
	unsigned int nodeid;
  unsigned int nthreads;
  unsigned int tracing_disabled;
  NewQueue_t *recv_queue;
  NewQueue_t *send_queue;
  struct thread_t *threads;
	unsigned virtual_threads;
} task_t;

typedef struct ptask_t
{
  unsigned int ntasks;
  struct task_t *tasks;
} ptask_t;

typedef struct appl_t
{
	unsigned int nptasks;
	struct ptask_t *ptasks;
} appl_t;

extern ptask_t *obj_table;

void InitializeObjectTable (unsigned num_appl, struct input_t * files,
	unsigned long nfiles);

#endif
