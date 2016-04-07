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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
#endif

#include "communicators.h"
#include "semantics.h"
#include "mpi_comunicadors.h"
#include "paraver_generator.h"
#include "dimemas_generator.h"

//#define DEBUG_COMMUNICATORS

/******************************************************************************
 ***  BuildInterCommunicatorFromFile
 ******************************************************************************/
static unsigned int BuildInterCommunicatorFromFile (event_t *current_event,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t * fset)
{
	unsigned foo;
	unsigned nevents = 1;
	uintptr_t comm1 = Get_EvComm (current_event);
	int leader1 = Get_EvTag (current_event);

	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(cpu);

	current_event = GetNextEvent_FS (fset, &foo, &ptask, &task, &thread);
	if (current_event != NULL)
	{
		uintptr_t comm2 = Get_EvComm (current_event);
		int leader2 = Get_EvTag (current_event);
		nevents++;

		current_event = GetNextEvent_FS (fset, &foo, &ptask, &task, &thread);

		if (current_event != NULL)
		{
			uintptr_t intercomm = Get_EvComm (current_event);
			nevents++;

#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "[DEBUG] Creating intercommunicator %lu from <%lu,%lu>\n",
			  intercomm, comm1, comm2);
#endif

#if defined(PARALLEL_MERGE)
			ParallelMerge_AddInterCommunicator (ptask, task, intercomm, comm1, leader1,
			  comm2, leader2);
#else
			addInterCommunicator (intercomm, comm1, leader1, comm2, leader2,
			  ptask, task);
#endif
		}
	}

	return nevents;
}


/******************************************************************************
 ***  BuildCommunicatorFromFile
 ******************************************************************************/
static unsigned int BuildCommunicatorFromFile (event_t *current_event,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t * fset)
{
	TipusComunicador new_comm;
	unsigned int i = 0;
	unsigned int foo;
	unsigned int EvType = Get_EvEvent (current_event);
	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(cpu);

	/* New communicator definition starts */
	new_comm.id = Get_EvComm (current_event);
	new_comm.num_tasks = Get_EvSize (current_event);
	new_comm.tasks = (int*) malloc(sizeof(int)*new_comm.num_tasks);
	if (NULL == new_comm.tasks)
	{
		fprintf (stderr, "mpi2prv: Can't allocate memory for a COMM SELF alias\n");
		fflush (stderr);
		exit (-1);
	}
#if defined(DEBUG_COMMUNICATORS)
	fprintf (stderr, "DEBUG: New comm: id=%lu, num_tasks=%u\n", new_comm.id, new_comm.num_tasks);
#endif

	/* Process each communicator member */
	current_event = GetNextEvent_FS (fset, &foo, &ptask, &task, &thread);
	if (current_event != NULL)
		EvType = Get_EvEvent (current_event);

	while (i < new_comm.num_tasks && current_event != NULL && 
        (EvType == MPI_RANK_CREACIO_COMM_EV || EvType == FLUSH_EV))
	{
		if (EvType == MPI_RANK_CREACIO_COMM_EV)
		{
			/* Save this task as member of the communicator */
			new_comm.tasks[i] = Get_EvValue (current_event);
#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "  -- task %d\n", new_comm.tasks[i]);
#endif
			i++;
		}
		if (i < new_comm.num_tasks)
		{
			current_event = GetNextEvent_FS (fset, &foo, &ptask, &task, &thread);
			if (current_event != NULL)
				EvType = Get_EvEvent (current_event);
		}
	}

	/* End of communicator definition. Assign an alias for this communicator. */
	if (i != new_comm.num_tasks)
	{
		unsigned long long tmp_time = 0;
		if (current_event != NULL) tmp_time = Get_EvTime(current_event);
		fprintf (stderr, "mpi2prv: Error: Incorrect communicator definition! (%d out of %d definitions)\n"
			"EvType: %u, Time: %llu, ptask: %u, task: %u, thread: %u\n",
			i, new_comm.num_tasks, EvType, tmp_time, ptask, task, thread);
		exit (0);
	}
	else
	{
#if defined(PARALLEL_MERGE)
		ParallelMerge_AddIntraCommunicator (ptask, task, 0, new_comm.id,
		  new_comm.num_tasks, new_comm.tasks);
#else
		afegir_comunicador (&new_comm, ptask, task);
#endif
	}

	free (new_comm.tasks);

	return i;
}

/******************************************************************************
 ***  GenerateAliesComunicator
 ******************************************************************************/
int GenerateAliesComunicator (
   event_t * current_event, unsigned long long current_time, unsigned int cpu,
   unsigned int ptask, unsigned int task, unsigned int thread, FileSet_t * fset,
   unsigned long long *num_events, int traceformat)
{
	unsigned int i = 0;
	unsigned int EvValue = Get_EvValue (current_event);
	unsigned int EvCommType = Get_EvTarget (current_event);
	unsigned int EvType = Get_EvEvent (current_event);

	if (EvValue == EVT_BEGIN)
	{
#if defined(DEBUG_COMMUNICATORS)
		fprintf (stderr, "DEBUG: new communicator definition (commtype = %d)\n",
			EvCommType);
#endif

		if (PRV_SEMANTICS == traceformat)
			if (Get_EvAux (current_event)) /* Shall we emit this into tracefile? */ 
			{
				trace_paraver_state (cpu, ptask, task, thread, current_time);
				trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EVT_BEGIN);
			}

		/* Build COMM WORLD communicator */
		if (MPI_COMM_WORLD_ALIAS == EvCommType)
		{
			TipusComunicador new_comm;
			unsigned int i;

#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "DEBUG: defining MPI_COMM_WORLD (%d members)\n", Get_EvSize (current_event));
#endif

			new_comm.id = Get_EvComm (current_event);
			new_comm.num_tasks = Get_EvSize (current_event);
			new_comm.tasks = (int*) malloc(sizeof(int)*new_comm.num_tasks);
			if (NULL == new_comm.tasks)
			{
				fprintf (stderr, "mpi2prv: Can't allocate memory for a COMM WORLD alias\n");
				fflush (stderr);
				exit (-1);
			}
			for (i = 0; i < new_comm.num_tasks; i++)
				new_comm.tasks[i] = i;
#if defined(PARALLEL_MERGE)
			ParallelMerge_AddIntraCommunicator (ptask, task, MPI_COMM_WORLD_ALIAS, new_comm.id, new_comm.num_tasks, new_comm.tasks);
#else
			afegir_comunicador (&new_comm, ptask, task);
#endif
			free (new_comm.tasks);
		}
		/* Build COMM SELF communicator */
		else if (MPI_COMM_SELF_ALIAS == EvCommType)
		{
			TipusComunicador new_comm;

#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "DEBUG: defining MPI_COMM_SELF (%d members)\n", Get_EvSize (current_event));
#endif

			new_comm.id = Get_EvComm (current_event);
			new_comm.num_tasks = 1;
			new_comm.tasks = (int*) malloc(sizeof(int)*new_comm.num_tasks);
			if (NULL == new_comm.tasks)
			{
				fprintf (stderr, "mpi2prv: Can't allocate memory for a COMM SELF alias\n");
				fflush (stderr);
				exit (-1);
			}
			new_comm.tasks[0] = task-1;
#if defined(PARALLEL_MERGE)
			ParallelMerge_AddIntraCommunicator (ptask, task, MPI_COMM_SELF_ALIAS, new_comm.id, new_comm.num_tasks, new_comm.tasks);
#else
			afegir_comunicador (&new_comm, ptask, task);
#endif
			free (new_comm.tasks);
		}
		else if (MPI_NEW_INTERCOMM_ALIAS == EvCommType)
		{
#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "DEBUG: defining new INTERCOMM\n");
#endif
			i = BuildInterCommunicatorFromFile (current_event, current_time, cpu, ptask, task, thread, fset);
		}
		else
		{
#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "DEBUG: defining new COMM (%d members id = %d)\n", Get_EvSize(current_event), Get_EvComm (current_event));
#endif

			i = BuildCommunicatorFromFile (current_event, current_time, cpu, ptask, task,
				thread, fset);
		}
	}
	else if (EvValue == EVT_END)
	{
		if (PRV_SEMANTICS == traceformat)
			if (Get_EvAux (current_event)) /* Shall we emit this into tracefile? */
				trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EVT_END);
	}

	*num_events = i+1;
	/* Count how many records have we processed
		(i communicator members + begin of communicator event) */
	return 0;
}
