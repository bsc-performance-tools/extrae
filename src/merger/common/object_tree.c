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

#include "object_tree.h"

ptask_t *obj_table = NULL;

/******************************************************************************
 ***  InitializeObjectTable
 ******************************************************************************/
void InitializeObjectTable (unsigned num_appl, struct input_t * files,
	unsigned long nfiles)
{
	unsigned int ptask, task, thread, i, j;
	unsigned int maxtasks = 0, maxthreads = 0;

	/* This is not the perfect way to allocate everything, but it's
	  good enough for runs where all the ptasks (usually 1), have the
	  same number of threads */

	for (i = 0; i < nfiles; i++)
	{
		maxtasks = MAX(files[i].task, maxtasks);
		maxthreads = MAX(files[i].thread, maxthreads);
	}

	obj_table = (ptask_t*) malloc (sizeof(ptask_t)*num_appl);
	if (NULL == obj_table)
	{
		fprintf (stderr, "mpi2prv: Error! Unable to alloc memory for %d ptasks!\n", num_appl);
		fflush (stderr);
		exit (-1);
	}
	for (i = 0; i < num_appl; i++)
	{
		/* Allocate per task information within each ptask */
		obj_table[i].tasks = (task_t*) malloc (sizeof(task_t)*maxtasks);
		if (NULL == obj_table[i].tasks)
		{
			fprintf (stderr, "mpi2prv: Error! Unable to alloc memory for %d tasks (ptask = %d)\n", maxtasks, i+1);
			fflush (stderr);
			exit (-1);
		}
		for (j = 0; j < maxtasks; j++)
		{
			/* Initialize pending communication queues for each task */
			CommunicationQueues_Init (&(obj_table[i].tasks[j].send_queue),
			  &(obj_table[i].tasks[j].recv_queue));

			/* Allocate per thread information within each task */
			obj_table[i].tasks[j].threads = (thread_t*) malloc (sizeof(thread_t)*maxthreads);
			if (NULL == obj_table[i].tasks[j].threads)
			{
				fprintf (stderr, "mpi2prv: Error! Unable to alloc memory for %d threads (ptask = %d / task = %d)\n", maxthreads, i+1, j+1);
				fflush (stderr);
				exit (-1);
			}
		}
	}

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	INIT_QUEUE (&CountersTraced);
#endif

	for (ptask = 0; ptask < num_appl; ptask++)
	{
		obj_table[ptask].ntasks = 0;
		for (task = 0; task < maxtasks; task++)
		{
			task_t *task_info = GET_TASK_INFO(ptask+1,task+1);
			task_info->tracing_disabled = FALSE;
			task_info->nthreads = 0;
			task_info->num_virtual_threads = 0;

			for (thread = 0; thread < maxthreads; thread++)
			{
				thread_t *thread_info = GET_THREAD_INFO(ptask+1,task+1,thread+1);

				thread_info->virtual_thread = thread+1;
				task_info->num_virtual_threads = MAX(thread_info->virtual_thread, task_info->num_virtual_threads);

				thread_info->nStates = 0;
				thread_info->First_Event = TRUE;
				thread_info->HWCChange_count = 0;
				thread_info->MatchingComms = TRUE;

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				thread_info->HWCSets = NULL;
				thread_info->HWCSets_types = NULL;
				thread_info->num_HWCSets = 0;
				thread_info->current_HWCSet = 0;
#endif
			}
		}
	}

	for (i = 0; i < nfiles; i++)
	{
		task_t *task_info = GET_TASK_INFO(files[i].ptask, files[i].task);
		thread_t *thread_info = GET_THREAD_INFO(files[i].ptask, files[i].task, files[i].thread);

		obj_table[ptask-1].ntasks = MAX (obj_table[ptask-1].ntasks, task);
		task_info->nodeid = files[i].nodeid;
		task_info->nthreads = MAX (task_info->nthreads, files[i].thread);
		thread_info->cpu = files[i].cpu;
		thread_info->dimemas_size = 0;
	}

	/* This is needed for get_option_merge_NanosTaskView() == FALSE */
	for (ptask = 0; ptask < num_appl; ptask++)
		for (task = 0; task < maxtasks; task++)
		{
			task_t *task_info = GET_TASK_INFO(ptask+1, task+1);
			task_info->num_active_task_threads = 0;
			task_info->active_task_threads = NULL;
		}
}

