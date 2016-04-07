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

appl_t ApplicationTable;

/******************************************************************************
 ***  InitializeObjectTable
 ******************************************************************************/
void InitializeObjectTable (unsigned num_appl, struct input_t * files,
	unsigned long nfiles)
{
	unsigned int ptask, task, thread, i, j, v;
	unsigned int ntasks[num_appl], **nthreads = NULL;

	/* First step, collect number of applications, number of tasks per application and
	   number of threads per task within an app */
	for (i = 0; i < num_appl; i++)
		ntasks[i] = 0;

	for (i = 0; i < nfiles; i++)
		ntasks[files[i].ptask-1] = MAX(files[i].task, ntasks[files[i].ptask-1]);

	nthreads = (unsigned**) malloc (num_appl*sizeof(unsigned*));
	ASSERT(nthreads!=NULL, "Cannot allocate memory to store nthreads for whole applications");

	for (i = 0; i < num_appl; i++)
	{
		nthreads[i] = (unsigned*) malloc (ntasks[i]*sizeof(unsigned));
		ASSERT(nthreads[i]!=NULL, "Cannot allocate memory to store nthreads for application");

		for (j = 0; j < ntasks[i]; j++)
			nthreads[i][j] = 0;
	}

	for (i = 0; i < nfiles; i++)
		nthreads[files[i].ptask-1][files[i].task-1] = MAX(files[i].thread, nthreads[files[i].ptask-1][files[i].task-1]);

	/* Second step, allocate structures respecting the number of apps, tasks and threads found */
	ApplicationTable.nptasks = num_appl;
	ApplicationTable.ptasks = (ptask_t*) malloc (sizeof(ptask_t)*num_appl);
	ASSERT(ApplicationTable.ptasks!=NULL, "Unable to allocate memory for ptasks");

	for (i = 0; i < ApplicationTable.nptasks; i++)
	{
		/* Allocate per task information within each ptask */
		ApplicationTable.ptasks[i].ntasks = ntasks[i];
		ApplicationTable.ptasks[i].tasks = (task_t*) malloc (sizeof(task_t)*ntasks[i]);
		ASSERT(ApplicationTable.ptasks[i].tasks!=NULL, "Unable to allocate memory for tasks");

		for (j = 0; j < ApplicationTable.ptasks[i].ntasks; j++)
		{
			/* Initialize pending communication queues for each task */
			CommunicationQueues_Init (
			  &(ApplicationTable.ptasks[i].tasks[j].send_queue),
			  &(ApplicationTable.ptasks[i].tasks[j].recv_queue));

			/* Allocate per thread information within each task */
			ApplicationTable.ptasks[i].tasks[j].threads = (thread_t*) malloc (sizeof(thread_t)*nthreads[i][j]);
			ASSERT(ApplicationTable.ptasks[i].tasks[j].threads!=NULL,"Unable to allocate memory for threads");
		}
	}

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	INIT_QUEUE (&CountersTraced);
#endif

	/* 3rd step, Initialize the object table structure */
	for (ptask = 0; ptask < ApplicationTable.nptasks; ptask++)
		for (task = 0; task < ApplicationTable.ptasks[ptask].ntasks; task++)
		{
			task_t *task_info = GET_TASK_INFO(ptask+1,task+1);
			task_info->tracing_disabled = FALSE;
			task_info->nthreads = nthreads[ptask][task];
			task_info->num_virtual_threads = nthreads[ptask][task];
			task_info->MatchingComms = TRUE;
			task_info->match_zone = 0;
			task_info->num_binary_objects = 0;
			task_info->binary_objects = NULL;
			task_info->thread_dependencies = ThreadDependency_create();
			task_info->AddressSpace = AddressSpace_create();

			for (thread = 0; thread < nthreads[ptask][task]; thread++)
			{
				thread_t *thread_info = GET_THREAD_INFO(ptask+1,task+1,thread+1);

				/* Look for the appropriate CPU for this ptask, task, thread */
				for (i = 0; i < nfiles; i++)
					if (files[i].ptask == ptask+1 &&
					    files[i].task == task+1 &&
					    files[i].thread == thread+1)
					{
						thread_info->cpu = files[i].cpu;
						break;
					}

				thread_info->dimemas_size = 0;
				thread_info->virtual_thread = thread+1;
				thread_info->State_Stack = NULL;
				thread_info->nStates = 0;
				thread_info->nStates_Allocated = 0;
				thread_info->First_Event = TRUE;
				thread_info->HWCChange_count = 0;
				for (v = 0; v < MAX_CALLERS; v++)
					thread_info->AddressSpace_calleraddresses[v] = 0;
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				thread_info->HWCSets = NULL;
				thread_info->HWCSets_types = NULL;
				thread_info->num_HWCSets = 0;
				thread_info->current_HWCSet = 0;
#endif
			}
		}

	/* 4th step Assign the node ID */
	for (i = 0; i < nfiles; i++)
	{
		task_t *task_info = GET_TASK_INFO(files[i].ptask, files[i].task);
		task_info->nodeid = files[i].nodeid;
	}

	/* This is needed for get_option_merge_NanosTaskView() == FALSE */
	for (ptask = 0; ptask < ApplicationTable.nptasks; ptask++)
		for (task = 0; task < ApplicationTable.ptasks[ptask].ntasks; task++)
		{
			task_t *task_info = GET_TASK_INFO(ptask+1, task+1);
			task_info->num_active_task_threads = 0;
			task_info->active_task_threads = NULL;
		}

	/* Clean up */
	if (nthreads != NULL)
	{
		for (i = 0; i < num_appl; i++)
			if (nthreads[i] != NULL)
				free (nthreads[i]);
		free (nthreads);		
	}
}

static void AddBinaryObjectInto (unsigned ptask, unsigned task,
	unsigned long long start, unsigned long long end, unsigned long long offset,
	char *binary)
{
	task_t *task_info = GET_TASK_INFO(ptask, task);
	unsigned found = FALSE, u;

	if (!file_exists(binary))
		return;

	for (u = 0; u < task_info->num_binary_objects && !found; u++)
		found = strcmp (task_info->binary_objects[u].module, binary) == 0;

	if (!found)
	{
		unsigned last_index = task_info->num_binary_objects;
		task_info->binary_objects = (binary_object_t*) realloc (
		  task_info->binary_objects,
		  (last_index+1) * sizeof(binary_object_t));
		if (task_info->binary_objects == NULL)
		{
			fprintf (stderr, "Fatal error! Cannot allocate memory for binary object!\n");
			exit (-1);
		}
		task_info->binary_objects[last_index].module = strdup (binary);
		task_info->binary_objects[last_index].start_address = start;
		task_info->binary_objects[last_index].end_address = end;
		task_info->binary_objects[last_index].offset = offset;
		task_info->binary_objects[last_index].index = last_index+1;

#if defined(HAVE_BFD)
		task_info->binary_objects[last_index].nDataSymbols = 0;
		task_info->binary_objects[last_index].dataSymbols = NULL;

		BFDmanager_loadBinary (binary,
		  &(task_info->binary_objects[last_index].bfdImage),
		  &(task_info->binary_objects[last_index].bfdSymbols),
		  &(task_info->binary_objects[last_index].nDataSymbols),
		  &(task_info->binary_objects[last_index].dataSymbols));
#endif

		task_info->num_binary_objects++;
	}
}

void ObjectTable_AddBinaryObject (int allobjects, unsigned ptask, unsigned task,
	unsigned long long start, unsigned long long end, unsigned long long offset,
	char *binary)
{
	if (allobjects)
	{
		unsigned _ptask, _task;
		for (_ptask = 1; _ptask <= ApplicationTable.nptasks; _ptask++)
			for (_task = 1; _task <= ApplicationTable.ptasks[_ptask].ntasks; _task++)
				AddBinaryObjectInto (_ptask, _task, start, end, offset, binary);
	}
	else
		AddBinaryObjectInto (ptask, task, start, end, offset, binary);
}

char * ObjectTable_GetBinaryObjectName (unsigned ptask, unsigned task)
{
	task_t *task_info = GET_TASK_INFO(ptask, task);

	if (task_info->num_binary_objects > 0)
		return task_info->binary_objects[0].module;
	else
		return NULL;
}

binary_object_t* ObjectTable_GetBinaryObjectAt (unsigned ptask, unsigned task, UINT64 address)
{
	task_t *task_info = GET_TASK_INFO(ptask, task);
	unsigned u;

	for (u = 0; u < task_info->num_binary_objects; u++)
		if (address >= task_info->binary_objects[u].start_address &&
		    address <= task_info->binary_objects[u].end_address)
			return &(task_info->binary_objects[u]);

	return NULL;
}

int ObjectTable_GetSymbolFromAddress (UINT64 address, unsigned ptask,
	unsigned task, char **symbol)
{
#if !defined(HAVE_BFD)
	UNREFERENCED_PARAMETER(address);
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(symbol);
	return FALSE;
#else
	unsigned a;
	task_t *task_info = GET_TASK_INFO(ptask, task);

#if defined(DEBUG)
	fprintf (stderr, "mpi2prv: DEBUG: ObjectTable_GetSymbolFromAddress (%llx, %u, %u, %p)\n",
	  address, ptask, task, symbol);
#endif

	/* For now, emit only data symbols for binary object 0 */
	for (a = 0; a < task_info->binary_objects[0].nDataSymbols; a++)
	{
		data_symbol_t *d = &task_info->binary_objects[0].dataSymbols[a];
		uint64_t addr_begin = (uint64_t) d->address;
		uint64_t addr_end = addr_begin + d->size;
		if (addr_begin <= address && address < addr_end)
		{
			*symbol = d->name;
			return TRUE;
		}
	}
	return FALSE;
#endif
}

#if defined(BFD_MANAGER_GENERATE_ADDRESSES)
void ObjectTable_dumpAddresses (FILE *fd, unsigned eventstart)
{
	unsigned _ptask, _task, _address;

	/* Temporary, just dump information for ptask 1.task 1 */
	/* Emitting the rest of ptask/task requires some changes in mpimpi2prv */

	for (_ptask = 1; _ptask <= 1 /* ApplicationTable.nptasks */; _ptask++)
		for (_task = 1; _task <= 1 /* ApplicationTable.ptasks[_ptask].ntasks */; _task++)
		{
			task_t *task_info = GET_TASK_INFO(_ptask, _task);

			fprintf (fd, "EVENT_TYPE\n");
			fprintf (fd, "0 %u Object addresses for task %u.%u\n", eventstart++, _ptask, _task);
			fprintf (fd, "VALUES\n");

			/* For now, emit only data symbols for binary object 0 */
			for (_address = 0; _address < task_info->binary_objects[0].nDataSymbols; _address++)
			{
				data_symbol_t *d = &task_info->binary_objects[0].dataSymbols[_address];

				fprintf (fd, "%u %s [0x%08llx-0x%08llx]\n",
				  _address+1,
				  d->name,
				  (unsigned long long) d->address, 
				  ((unsigned long long) d->address)+d->size-1);
			}
			fprintf (fd, "\n");
		}
}
#endif

