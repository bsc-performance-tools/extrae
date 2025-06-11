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

#ifndef __OBJECT_TREE_H__
#define __OBJECT_TREE_H__

#include "common.h"

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "file_set.h"
#include "new-queue.h"
#include "HardwareCounters.h"
#include "communication_queues.h"
#include "stack.h"
#include "thread_dependencies.h"
#include "address_space.h"
#if defined(HAVE_LIBADDR2LINE)
# include "addr2line.h"
# include "symtab.h"
#endif

#define MAX_STATES_ALLOCATION  128
#define MAX_CALLERS            100

/**
 * thread_t
 * 
 * This structure contains the information at the thread-level in the object tree
 */
typedef struct thread_st
{
	/* Where is this thread running? */
	unsigned int cpu;

	/* Did we passed the first event? */
	unsigned int First_Event;

	/* Count the number of HWC changes */
	unsigned int HWCChange_count;

	/* Information of the stack */
	int *State_Stack;
	int nStates;
	int nStates_Allocated;

	/* Stores the time of the last event */
	UINT64 Previous_Event_Time;

	/* Information of the last state record */
	paraver_rec_t incomplete_state_record;
	off_t incomplete_state_offset;

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	unsigned long long last_hw_group_change;
	hwc_id_t **HWCSets;

	int num_HWCSets;
	int current_HWCSet;
	long long counters[MAX_HWC];     /* HWC values */
#endif

	event_t *Send_Rec;               /* Store send records */
	event_t *Recv_Rec;               /* Store receive records */
	FileItem_t *file;

	unsigned long long dimemas_size; /* Store dimemas translation size */

	unsigned virtual_thread;         /* if so, which virtual thread is? */
	unsigned active_task_thread;     /* if so, which active task has been resumed? */

	/* Address space preparation */
	uint64_t AddressSpace_size;
	uint64_t AddressSpace_timeCreation;
	uint64_t AddressSpace_calleraddresses[MAX_CALLERS];
	uint32_t AddressSpace_callertype;
} thread_t;

typedef struct active_task_thread_stack_type_st
{
	mpi2prv_stack_t *stack;
	unsigned type;
} active_task_thread_stack_type_t;

typedef struct active_task_thread_st
{
	active_task_thread_stack_type_t *stacked_type;
	unsigned num_stacks;
} active_task_thread_t;

/**
 * task_t
 * 
 * This structure contains the information at the task-level in the object tree
 */
typedef struct task_st
{
	unsigned int nodeid;
	unsigned int nthreads;
	thread_t *threads;
	unsigned int tracing_disabled;

	/* Controls whether this task matches comms or not */
	int MatchingComms;
	int match_zone;
	NewQueue_t *recv_queue;
	NewQueue_t *send_queue;

	/* Arrangement of thread dependencies within the task level */
	struct ThreadDependencies_st * thread_dependencies;

	/* Address space variables & preparation info */
	struct AddressSpace_st *AddressSpace;

	unsigned num_virtual_threads;
	unsigned num_active_task_threads;
	active_task_thread_t *active_task_threads;

	/* Address translation */
	char *proc_self_exe;    // Name of the main binary taken from /proc/self/exe
#if defined(HAVE_LIBADDR2LINE)
	maps_t *proc_self_maps; // Dump of the /proc/self/maps for address translation
	addr2line_t *addr2line; // Handler for libaddr2line
#endif
} task_t;

/**
 * ptask_t
 * 
 * This structure contains the information at the application-level in the object tree
 */
typedef struct ptask_st
{
  unsigned int ntasks;
  task_t *tasks;
} ptask_t;

/**
 * appl_t
 * 
 * This is the top-level structure of the object tree, and contains the information of all applications being merged
 */
typedef struct appl_st
{
	unsigned int nptasks;
	ptask_t *ptasks;
} appl_t;

// ApplicationTable is a global instance of the object tree
extern appl_t ApplicationTable;

// Macros to access the object tree

#define ObjectTree_getNumPtasks() \
	(ApplicationTable.nptasks)

#define ObjectTree_getNumTasks(ptask) \
	(ApplicationTable.ptasks[ptask - 1].ntasks)

#define ObjectTree_getPtaskInfo(ptask) \
	&(ApplicationTable.ptasks[ptask - 1])

#define ObjectTree_getTaskInfo(ptask, task) \
	&(ApplicationTable.ptasks[ptask - 1].tasks[task - 1])

#define ObjectTree_getThreadInfo(ptask, task, thread) \
	&(ApplicationTable.ptasks[ptask - 1].tasks[task - 1].threads[thread - 1])


// Functions to access the object tree

void   ObjectTree_Initialize (int worker_id, unsigned num_appl, struct input_t * files, unsigned long nfiles);

void   ObjectTree_setProcMaps(unsigned ptask, unsigned task, char *maps_file);
void   ObjectTree_setProcExe(unsigned ptask, unsigned task, char *self_exe);

char * ObjectTree_getMainBinary(unsigned ptask, unsigned task);

#endif /* __OBJECT_TREE_H__ */
