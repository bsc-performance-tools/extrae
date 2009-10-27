/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/HardwareCounters.c,v $
 | 
 | @last_commit: $Date: 2009/01/12 16:16:36 $
 | @version:     $Revision: 1.13 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: HardwareCounters.c,v 1.13 2009/01/12 16:16:36 harald Exp $";

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif

#include "queue.h"
#include "events.h"
#include "object_tree.h"
#include "HardwareCounters.h"
#include "trace_to_prv.h"
#include "paraver_generator.h"
#include "dimemas_generator.h"

/*
 * FreeListItems  : Free CntQueue queue items list.
 * CountersTraced : Queue of CntQueue strucutures. Will contain a list of all
 *                  the conunters that has been traced during application
 *                  execution.
 */
CntQueue *FreeListItems = NULL;
CntQueue CountersTraced;

/******************************************************************************
 **      Function name : HardwareCounters_Emit
 **      
 **      Description : 
 ******************************************************************************/

void HardwareCounters_Emit (int cpu, int ptask, int task, int thread,
	long long time, event_t * Event, unsigned int *outtype,
	unsigned long long *outvalue)
{
#warning "Aixo es forsa arriscat, cal que la crida tingui alocatat prou espai :S"
  int cnt;
  struct thread_t *Sthread;

  Sthread = &(obj_table[ptask-1].tasks[task-1].threads[thread-1]);

  for (cnt = 0; cnt < MAX_HWC; cnt++)
  {
    /* If using PAPI, they can be stored in absolute or relative manner,
    depending if sampling was activated or not */
#if defined(PAPI_COUNTERS)
# if defined(SAMPLING_SUPPORT)
    if (Sthread->counterEvents[cnt] != NO_COUNTER &&
		    Sthread->counterEvents[cnt] != SAMPLE_COUNTER)
# else
    if (Sthread->counterEvents[cnt] != NO_COUNTER)
# endif
    {
      /*
       * Si els comptadors acumulen cal aixo:
       * *   value = Event->HWCValues[ cnt ]-Sthread->counters[cnt];
       * * pero com que faig un reset a cada read: 
       */
# if defined(SAMPLING_SUPPORT)
			/* Protect when counters are incorrect (major timestamp, lower counter value) */
			if (Event->HWCValues[cnt] >= Sthread->counters[cnt])
			{
				outvalue[cnt] = Event->HWCValues[cnt] - Sthread->counters[cnt];
				outtype[cnt] = HWC_COUNTER_TYPE (Sthread->counterEvents[cnt]);
			}
			else
			{
				outtype[cnt] = NO_COUNTER;
			}
# else
      outvalue[cnt] = Event->HWCValues[cnt];
      outtype[cnt] = HWC_COUNTER_TYPE (Sthread->counterEvents[cnt]);
# endif

      Sthread->counters[cnt] = Event->HWCValues[cnt];
    }
		else
			outtype[cnt] = NO_COUNTER;

#elif defined(PMAPI_COUNTERS)

		if (Sthread->counterEvents[cnt] != NO_COUNTER)
		{
			outvalue[cnt]= Event->HWCValues[cnt] - Sthread->counters[cnt];
			outtype[cnt] = HWC_COUNTER_TYPE (cnt, Sthread->counterEvents[cnt]);
			Sthread->counters[cnt] = Event->HWCValues[cnt];
		}
		else
			outtype[cnt] = NO_COUNTER;

#endif

  }
}

static int HardwareCounters_Compare (long long *HWC1, int *used1, long long *HWC2, int *used2)
{
	int i;

	for (i = 0; i < MAX_HWC; i++)
		if ((HWC1[i] != HWC2[i]) || (used1[i] != used2[i]))
			return FALSE;

	return TRUE;
}

static int HardwareCounters_Exist (long long *HWC, int *used)
{
	CntQueue *queue, *ptmp;

	queue = &CountersTraced;
	for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev)
		if (HardwareCounters_Compare (ptmp->Events, ptmp->Traced, HWC, used))
			return TRUE;

	return FALSE;
}

void HardwareCounters_Show (event_t * Event)
{
  int cnt;
  fprintf (stdout, "COUNTERS: ");
  for (cnt = 0; cnt < MAX_HWC; cnt++)
    fprintf (stdout, "[%llu] ", Event->HWCValues[cnt]);
  fprintf (stdout, "\n");
}

void HardwareCounters_Get (event_t *Event, unsigned long long *buffer)
{
	int cnt;
	for (cnt = 0; cnt < MAX_HWC; cnt++)
		buffer[cnt] = Event->HWCValues[cnt];
}

/******************************************************************************
 **      Function name : HardwareCounters_Change
 **      
 **      Description : 
 ******************************************************************************/

void HardwareCounters_Change (int cpu, int ptask, int task, int thread,
	event_t *current, unsigned long long time, unsigned int *outtypes,
	unsigned long long *outvalues)
{
#warning "Aixo es forsa arriscat, cal que la crida tingui alocatat prou espai :S"
	int cnt;
	CntQueue *cItem;
	struct thread_t *Sthread = &(obj_table[ptask-1].tasks[task-1].threads[thread-1]);
	int counters_used[MAX_HWC];

	for (cnt = 0; cnt < MAX_HWC; cnt++)
		counters_used[cnt] = (current->HWCValues[cnt] != NO_COUNTER);

	outtypes[0] = HWC_GROUP_ID; outvalues[0] = 1+Get_EvValue (current);

	if (Sthread->First_HWCChange)
	{
		for (cnt = 0; cnt < MAX_HWC; cnt++)
		{
			Sthread->counterEvents[cnt] = current->HWCValues[cnt];
			Sthread->counters[cnt] = 0;
			outtypes[cnt+1] = NO_COUNTER;
		}
		Sthread->First_HWCChange = FALSE;
	}
	else
	{
		for (cnt = 0; cnt < MAX_HWC; cnt++)
		{
			Sthread->counterEvents[cnt] = current->HWCValues[cnt];
			Sthread->counters[cnt] = 0;

			/* Emit counters with value 0 at the very beginning*/
			if (counters_used[cnt])
			{
#if defined(PMAPI_COUNTERS)
				outtypes[cnt+1] = HWC_COUNTER_TYPE(cnt, Sthread->counterEvents[cnt]);
#else
				outtypes[cnt+1] = HWC_COUNTER_TYPE(Sthread->counterEvents[cnt]);
#endif
				outvalues[cnt+1] = 0;
			}
			else
				outtypes[cnt+1] = NO_COUNTER;
		}
	}

	/* Add this counters (if didn't exist) to a queue in order to put them into the PCF */
	if (HardwareCounters_Exist (current->HWCValues, counters_used))
		return;

	ALLOC_NEW_ITEM (FreeListItems, sizeof (CntQueue), cItem, "CntQueue");
	for (cnt = 0; cnt < MAX_HWC; cnt++)
	{
		cItem->Events[cnt] = current->HWCValues[cnt];
		cItem->Traced[cnt] = (current->HWCValues[cnt] != NO_COUNTER);
	}
  ENQUEUE_ITEM (&CountersTraced, cItem);
}

void HardwareCounters_SetOverflow (int ptask, int task, int thread, event_t *Event)
{
  int cnt;
  struct thread_t *Sthread = &(obj_table[ptask-1].tasks[task-1].threads[thread-1]);

  for (cnt = 0; cnt < MAX_HWC; cnt++)
		if (Event->HWCValues[cnt] == SAMPLE_COUNTER)
			Sthread->counterEvents[cnt] = SAMPLE_COUNTER;
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-tags.h"
#include "mpi-aux.h"

static void HardwareCounters_Add (long long *HWCValues, int *used)
{
  int cnt;
  CntQueue *cItem;
	if (HardwareCounters_Exist(HWCValues, used))
		return;

  ALLOC_NEW_ITEM (FreeListItems, sizeof (CntQueue), cItem, "CntQueue");
  for (cnt = 0; cnt < MAX_HWC; cnt++)
  {
    cItem->Events[cnt] = HWCValues[cnt];
		cItem->Traced[cnt] = used[cnt];
  }
  ENQUEUE_ITEM (&CountersTraced, cItem);
}

void Share_Counters_Usage (int size, int rank)
{
	int res;
	MPI_Status s;

	if (rank == 0)
	{
		/* Code to run the master */
		int slave, ncounters;
		long long counters[MAX_HWC];
		int used[MAX_HWC];

		for (slave = 1; slave < size; slave++)
		{
			/* How many set of counters has each slave? */
			res = MPI_Recv (&ncounters, 1, MPI_INTEGER, slave, NUMBER_OF_HWC_SETS_TAG, MPI_COMM_WORLD, &s);
			MPI_CHECK(res, MPI_Recv, "Receiving number of sets of HWC");

			res = MPI_Send (&ncounters, 1, MPI_INTEGER, slave, HWC_SETS_READY, MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Send, "Sending ready statement");

			if (ncounters > 0)
			{
				int i;
				/* Just receive the counters of each slave */
				for (i = 0; i < ncounters; i++)
				{
					res = MPI_Recv (counters, MAX_HWC, MPI_LONG_LONG, slave, HWC_SETS_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Receiving HWC");
					res = MPI_Recv (used, MAX_HWC, MPI_INTEGER, slave, HWC_SETS_ENABLED_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Receiving used HWC bitmap");
					HardwareCounters_Add (counters, used);
				}
			}
		}
	}
	else
	{
		/* Code to run the slaves */
		/* Gather all HWC info, and send to the master */
		int count;
		CntQueue *queue, *ptmp;

		count = 0;
		queue = &CountersTraced;
		for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev)
			count++;

		res = MPI_Send (&count, 1, MPI_INTEGER, 0, NUMBER_OF_HWC_SETS_TAG, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Send, "Sending number of HWC sets");

		res = MPI_Recv (&count, 1, MPI_INTEGER, 0, HWC_SETS_READY, MPI_COMM_WORLD, &s);
		MPI_CHECK(res, MPI_Recv, "Receiving ready statement");

		if (count > 0)
		{
  		queue = &CountersTraced;
 		 	for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev)
			{
				res = MPI_Send (ptmp->Events, MAX_HWC, MPI_LONG_LONG, 0, HWC_SETS_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Sending HWC");
				res = MPI_Send (ptmp->Traced, MAX_HWC, MPI_INTEGER, 0, HWC_SETS_ENABLED_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Sending used HWC bitmap");
			}
		}
	}
}
#endif

#endif /* USE_HARDWARE_COUNTERS  || HETEROGENEOUS_SUPPORT*/


