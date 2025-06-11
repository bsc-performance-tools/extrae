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

#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include "object_tree.h"
#include "utils.h"
#include "debug.h"
#include "xalloc.h"
#include "options.h"
#if defined(HAVE_LIBADDR2LINE)
# include "addr2info.h"
# include "maps.h"
#endif

appl_t ApplicationTable;


/**
 * getMapsFromMpit
 *
 * This function takes a file name with ".mpit" extension and replaces it with ".maps".A
 *
 * @param mpit_file The input file name with ".mpit" extension
 * @return A new string with the ".maps" extension. The caller is responsible for freeing the returned string.
 */
static char * getMapsFromMpit(char *mpit_file)
{
	char *maps_file = strdup(mpit_file);
	char *dot = strrchr(maps_file, '.');
	if (dot) {
		strcpy(dot, EXT_MAPS);
	}
	return maps_file;
}


/**
 * ObjectTree_Initialize
 *
 * Initializes the object tree structure according to the application's object hierarchy,
 * which includes the number of applications, tasks per application, and threads per task.
 * Each node in the tree stores information relevant to its corresponding level-application, task, or thread.
 *
 * @param worker_id The ID of the current parallel merger process
 * @param num_appl The number of applications
 * @param files The input_t structure containing all trace files
 * @param nfiles The number of trace files
 */
void ObjectTree_Initialize (int worker_id, unsigned num_appl, struct input_t * files, unsigned long nfiles)
{
	unsigned int ptask, task, thread, i, j, v;
	unsigned int ntasks[num_appl], **nthreads = NULL;

	/*
	 * 1st step, collect number of applications, number of tasks per application and
	 * number of threads per task within an app
	 */
	for (i = 0; i < num_appl; i++) {
		ntasks[i] = 0;
	}

	for (i = 0; i < nfiles; i++) {
		ntasks[files[i].ptask-1] = MAX(files[i].task, ntasks[files[i].ptask-1]);
	}

	nthreads = (unsigned**) xmalloc (num_appl*sizeof(unsigned*));
	for (i = 0; i < num_appl; i++) 
	{
		nthreads[i] = (unsigned*) xmalloc (ntasks[i]*sizeof(unsigned));
		
		for (j = 0; j < ntasks[i]; j++) {
			nthreads[i][j] = 0;
		}
	}
	for (i = 0; i < nfiles; i++) {
		nthreads[files[i].ptask-1][files[i].task-1] = MAX(files[i].thread, nthreads[files[i].ptask-1][files[i].task-1]);
	}

	/* 2nd step, allocate structures for the number of apps, tasks and threads found */
	ApplicationTable.nptasks = num_appl;
	ApplicationTable.ptasks = (ptask_t*) xmalloc (sizeof(ptask_t)*num_appl);
	
	for (i = 0; i < ApplicationTable.nptasks; i++)
	{
		// Allocate per task information within each ptask
		ApplicationTable.ptasks[i].ntasks = ntasks[i];
		ApplicationTable.ptasks[i].tasks = (task_t*) xmalloc (sizeof(task_t)*ntasks[i]);
		
		for (j = 0; j < ApplicationTable.ptasks[i].ntasks; j++)
		{
			// Initialize pending communication queues for each task
			CommunicationQueues_Init (
			  &(ApplicationTable.ptasks[i].tasks[j].send_queue),
			  &(ApplicationTable.ptasks[i].tasks[j].recv_queue));

			// Allocate per thread information within each task
			ApplicationTable.ptasks[i].tasks[j].threads = (thread_t*) xmalloc (sizeof(thread_t)*nthreads[i][j]);
		}
	}

	/* 3rd step, initialize the corresponding object-level structures */
	for (ptask = 0; ptask < ApplicationTable.nptasks; ptask++) 
	{
		for (task = 0; task < ApplicationTable.ptasks[ptask].ntasks; task++)
		{
			// Initialize task-level structures
			task_t *task_info = ObjectTree_getTaskInfo(ptask+1,task+1);
			task_info->tracing_disabled = FALSE;
			task_info->nthreads = nthreads[ptask][task];
			task_info->num_virtual_threads = nthreads[ptask][task];
			task_info->MatchingComms = TRUE;
			task_info->match_zone = 0;
			task_info->thread_dependencies = ThreadDependency_create();
			task_info->AddressSpace = AddressSpace_create();
			// Initialize structures related to address translations
			task_info->proc_self_exe = NULL;
#if defined(HAVE_LIBADDR2LINE)
			task_info->proc_self_maps = NULL;
			task_info->addr2line = NULL;
#endif

			// Initialize thread-level structures
			for (thread = 0; thread < nthreads[ptask][task]; thread++)
			{
				thread_t *thread_info = ObjectTree_getThreadInfo(ptask+1,task+1,thread+1);

				// Look for the appropriate CPU for this ptask, task, thread
				for (i = 0; i < nfiles; i++) 
				{
					if (files[i].ptask == ptask+1 &&
					    files[i].task == task+1 &&
					    files[i].thread == thread+1)
					{
						thread_info->cpu = files[i].cpu;
						break;
					}
				}

				thread_info->dimemas_size = 0;
				thread_info->virtual_thread = thread+1;
				thread_info->State_Stack = NULL;
				thread_info->nStates = 0;
				thread_info->nStates_Allocated = 0;
				thread_info->First_Event = TRUE;
				thread_info->HWCChange_count = 0;
				for (v = 0; v < MAX_CALLERS; v++) {
					thread_info->AddressSpace_calleraddresses[v] = 0;
				}
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				thread_info->HWCSets = NULL;
				thread_info->num_HWCSets = 0;
				thread_info->current_HWCSet = 0;
#endif
			}
		}
	}

	/* 4th step, assign the node ID */
	for (i = 0; i < nfiles; i++)
	{
		task_t *task_info = ObjectTree_getTaskInfo(files[i].ptask, files[i].task);
		task_info->nodeid = files[i].nodeid;
	}

	/* This is needed for get_option_merge_NanosTaskView() == FALSE */
	for (ptask = 0; ptask < ApplicationTable.nptasks; ptask++) 
	{
		for (task = 0; task < ApplicationTable.ptasks[ptask].ntasks; task++)
		{
			task_t *task_info = ObjectTree_getTaskInfo(ptask+1, task+1);
			task_info->num_active_task_threads = 0;
			task_info->active_task_threads = NULL;
		}
	}

#if defined(HAVE_LIBADDR2LINE)
	/* Load the maps files for address translations */
	if (worker_id == 0) // For now, the translations are done in the master merger only. In the future remove this constraint.
	{
		for (i = 0; i < nfiles; i++)
		{
			if (files[i].thread == 1) // Only for the main thread of each task
			{
				// Assign to each task its corresponding maps file
				ObjectTree_setProcMaps(files[i].ptask, files[i].task, getMapsFromMpit(files[i].name));
			}
		}
	}
#endif

        /*
         * Initialization of the address translation module 
	 *
	 * This operation is performed without protection to allow setup of the
	 * tables that store user-translated addresses based on O|P|U entries
	 * in the SYM file.
	 *
	 * Previously, this method initialized both the tables holding already
	 * translated addresses by category (i.e. MPI call stack, OMP outlined,
	 * CUDA kernel, etc.), as well as the translation back-end. The back-end
	 * initialization has since been removed from here. It is now performed
	 * lazily the first time an address translation is attempted.
	 * See Initialize_Translation_Backend at addr2info.c for details.
         */
	Address2Info_Initialize();

	/* Clean up */
	if (nthreads != NULL)
	{
		for (i = 0; i < num_appl; i++) 
		{
			if (nthreads[i] != NULL) {
				xfree (nthreads[i]);
			}
		}
		xfree (nthreads);
	}
}


#if defined(HAVE_LIBADDR2LINE)
/**
 * ObjectTree_setProcMaps
 * 
 * This function sets the proc_self_maps for a given task in the object tree.
 * 
 * @param ptask The ptask ID
 * @param task The task ID
 * @param maps_file The path to the maps file
 */
void ObjectTree_setProcMaps(unsigned ptask, unsigned task, char *maps_file)
{
	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);

	if (task_info && maps_file) {
		task_info->proc_self_maps = maps_parse_file(maps_file, OPTION_READ_SYMTAB);
	}
}
#endif


/**
 * ObjectTree_setProcExe
 * 
 * This function sets the proc_self_exe for a given task in the object tree.
 * 
 * @param ptask The ptask ID
 * @param task The task ID
 * @param self_exe The path to the executable file
 */
void ObjectTree_setProcExe(unsigned ptask, unsigned task, char *self_exe)
{
	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);

	if (task_info && self_exe) {
		task_info->proc_self_exe = strdup(self_exe);
	}
}


/**
 * ObjectTree_getMainBinary
 *
 * Gets the main binary name for the given ptask and task.
 * This function prioritizes to retrieve the binary name from the /proc/self/exe that was captured during the tracing (and stored in SYM entry 'X').
 * If that is not available, it checks for the user-given executable name (through mpi2prv flag -e).
 * If neither is available, it returns NULL to indicate that the binary name could not be determined.
 *
 * @param ptask The ptask ID
 * @param task The task ID
 * @return The main binary name for the given ptask and task, or NULL if not available
 */
char * ObjectTree_getMainBinary(unsigned ptask, unsigned task)
{
	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);
	if (task_info->proc_self_exe != NULL) {
		// Return the /proc/self/exe path that was captured during the tracing, if available
		return task_info->proc_self_exe;
	}
	char *user_given_exe = get_merge_ExecutableFileName();
	if (strlen(user_given_exe) > 0) {
		// Alternatively, return the user executable name, if given
		return user_given_exe;
	}
	return NULL;
}

