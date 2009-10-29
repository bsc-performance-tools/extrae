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

#ifndef _OBJECT_TREE_H
#define _OBJECT_TREE_H

#include "common.h"

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "file_set.h"
#include "HardwareCounters.h"

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
	unsigned int First_Event;

	/* Did we processed the firt HWC change event? */
	unsigned int First_HWCChange;

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
	int **HWCSets;
	int num_HWCSets;
	int current_HWCSet;
	long long counters[MAX_HWC];  /* HWC values */
#endif

	event_t *Send_Rec;  /* Store send records */
	event_t *Recv_Rec;  /* Store receive records */
	FileItem_t *file;

	unsigned long long dimemas_size; /* Store dimemas translation size */

} thread_t;

typedef struct task_t
{
	unsigned int nodeid;
  unsigned int nthreads;
  unsigned int tracing_disabled;
  struct thread_t *threads;
} task_t;

typedef struct ptask_t
{
  unsigned int ntasks;
  struct task_t *tasks;
} ptask_t;

#endif
