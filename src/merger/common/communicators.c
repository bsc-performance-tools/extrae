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
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

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

/******************************************************************************
 ***  BuildCommunicator
 ******************************************************************************/
static unsigned int BuildCommunicatorFromFile (event_t *current_event,
  unsigned long long current_time, unsigned int cpu, unsigned int ptask,
  unsigned int task, unsigned int thread, FileSet_t * fset)
{
	TipusComunicador nou_com;
	unsigned int i = 0;
	unsigned int foo;
	unsigned int EvType = Get_EvEvent (current_event);
	UNREFERENCED_PARAMETER(current_time);
	UNREFERENCED_PARAMETER(cpu);

	/* New communicator definition starts */
	nou_com.id = Get_EvComm (current_event);
	nou_com.num_tasks = Get_EvSize (current_event);
	nou_com.tasks = (int*) malloc(sizeof(int)*nou_com.num_tasks);
	if (NULL == nou_com.tasks)
	{
		fprintf (stderr, "mpi2prv: Can't allocate memory for a COMM SELF alias\n");
		fflush (stderr);
		exit (-1);
	}
#if defined(DEBUG_COMMUNICATORS)
	fprintf (stderr, "DEBUG: New comm: id=%d, num_tasks=%d\n", nou_com.id, nou_com.num_tasks);
#endif

	/* Process each communicator member */
	current_event = GetNextEvent_FS (fset, &foo, &ptask, &task, &thread);
	if (current_event != NULL)
		EvType = Get_EvEvent (current_event);

	while (i < nou_com.num_tasks && current_event != NULL && 
        (EvType == MPI_RANK_CREACIO_COMM_EV || EvType == FLUSH_EV))
	{
		if (EvType == MPI_RANK_CREACIO_COMM_EV)
		{
			/* Save this task as member of the communicator */
			nou_com.tasks[i] = Get_EvValue (current_event);
#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "  -- task %d\n", nou_com.tasks[i]);
#endif
			i++;
		}
		current_event = GetNextEvent_FS (fset, &foo, &ptask, &task, &thread);
		if (current_event != NULL)
			EvType = Get_EvEvent (current_event);
	}

	/* End of communicator definition. Assign an alias for this communicator. */
	if (i != nou_com.num_tasks)
	{
		fprintf (stderr, "mpi2prv: Error: Incorrect communicator definition!\n");
		exit (0);
	}
	else
	{
#if defined(PARALLEL_MERGE)
		AddCommunicator (ptask, task, 0, nou_com.id, nou_com.num_tasks, nou_com.tasks);
#else
		afegir_comunicador (&nou_com, ptask, task);
#endif
	}

	free (nou_com.tasks);

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

		/* Build COMM WORLD communicator */
		if (MPI_COMM_WORLD_ALIAS == EvCommType)
		{
			TipusComunicador nou_com;
			unsigned int i;

#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "DEBUG: defining MPI_COMM_WORLD (%d members)\n", Get_EvSize (current_event));
#endif

			nou_com.id = Get_EvComm (current_event);
			nou_com.num_tasks = Get_EvSize (current_event);
			nou_com.tasks = (int*) malloc(sizeof(int)*nou_com.num_tasks);
			if (NULL == nou_com.tasks)
			{
				fprintf (stderr, "mpi2prv: Can't allocate memory for a COMM WORLD alias\n");
				fflush (stderr);
				exit (-1);
			}
			for (i = 0; i < nou_com.num_tasks; i++)
				nou_com.tasks[i] = i;
#if defined(PARALLEL_MERGE)
			AddCommunicator (ptask, task, MPI_COMM_WORLD_ALIAS, nou_com.id, nou_com.num_tasks, nou_com.tasks);
#else
			afegir_comunicador (&nou_com, ptask, task);
#endif
			free (nou_com.tasks);
		}
		/* Build COMM SELF communicator */
		else if (MPI_COMM_SELF_ALIAS == EvCommType)
		{
			TipusComunicador nou_com;

#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "DEBUG: defining MPI_COMM_SELF (%d members)\n", Get_EvSize (current_event));
#endif

			nou_com.id = Get_EvComm (current_event);
			nou_com.num_tasks = 1;
			nou_com.tasks = (int*) malloc(sizeof(int)*nou_com.num_tasks);
			if (NULL == nou_com.tasks)
			{
				fprintf (stderr, "mpi2prv: Can't allocate memory for a COMM SELF alias\n");
				fflush (stderr);
				exit (-1);
			}
			nou_com.tasks[0] = task-1;
#if defined(PARALLEL_MERGE)
			AddCommunicator (ptask, task, MPI_COMM_SELF_ALIAS, nou_com.id, nou_com.num_tasks, nou_com.tasks);
#else
			afegir_comunicador (&nou_com, ptask, task);
#endif
			free (nou_com.tasks);
		}
		else
		{
#if defined(DEBUG_COMMUNICATORS)
			fprintf (stderr, "DEBUG: defining new COMM (%d members id = %d)\n", Get_EvSize(current_event), Get_EvComm (current_event));
#endif

			if (PRV_SEMANTICS == traceformat)
			{
				trace_paraver_state (cpu, ptask, task, thread, current_time);
				trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EVT_BEGIN);
			}

			i = BuildCommunicatorFromFile (current_event, current_time, cpu, ptask, task,
				thread, fset);

			if (PRV_SEMANTICS == traceformat)
				trace_paraver_event (cpu, ptask, task, thread, current_time, EvType, EVT_END);
		}
	}

	*num_events = i+1;
	/* Count how many records have we processed
		(i communicator members + begin of communicator event) */
	return 0;
}
