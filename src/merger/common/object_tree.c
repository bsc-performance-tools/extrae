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

	obj_table = (struct ptask_t*) malloc (sizeof(struct ptask_t)*num_appl);
	if (NULL == obj_table)
	{
		fprintf (stderr, "mpi2prv: Error! Unable to alloc memory for %d ptasks!\n", num_appl);
		fflush (stderr);
		exit (-1);
	}
	for (i = 0; i < num_appl; i++)
	{
		/* Allocate per task information within each ptask */
		obj_table[i].tasks = (struct task_t*) malloc (sizeof(struct task_t)*maxtasks);
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
			obj_table[i].tasks[j].threads = (struct thread_t*) malloc (sizeof(struct thread_t)*maxthreads);
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
			obj_table[ptask].tasks[task].tracing_disabled = FALSE;
			obj_table[ptask].tasks[task].nthreads = 0;
			for (thread = 0; thread < maxthreads; thread++)
			{
				obj_table[ptask].tasks[task].threads[thread].virtual_thread = thread+1;

				obj_table[ptask].tasks[task].threads[thread].nStates = 0;
				obj_table[ptask].tasks[task].threads[thread].First_Event = TRUE;
				obj_table[ptask].tasks[task].threads[thread].HWCChange_count = 0;
				obj_table[ptask].tasks[task].threads[thread].MatchingComms = TRUE;

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
				obj_table[ptask].tasks[task].threads[thread].HWCSets = NULL;
				obj_table[ptask].tasks[task].threads[thread].num_HWCSets = 0;
				obj_table[ptask].tasks[task].threads[thread].current_HWCSet = 0;
#endif
			}
		}
	}

	for (i = 0; i < nfiles; i++)
	{
		ptask = files[i].ptask;
		task = files[i].task;
		thread = files[i].thread;

		obj_table[ptask-1].tasks[task-1].nodeid = files[i].nodeid;
		obj_table[ptask-1].tasks[task-1].threads[thread-1].cpu = files[i].cpu;
		obj_table[ptask-1].tasks[task-1].threads[thread-1].dimemas_size = 0;
		obj_table[ptask-1].ntasks = MAX (obj_table[ptask-1].ntasks, task);
		obj_table[ptask-1].tasks[task-1].virtual_threads =
			obj_table[ptask-1].tasks[task-1].nthreads =
				MAX (obj_table[ptask-1].tasks[task-1].nthreads, thread);
	}
}
