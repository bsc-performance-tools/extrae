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
#include "utils.h"

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

int HardwareCounters_Emit (int ptask, int task, int thread,
	unsigned long long time, event_t * Event, int *outtype,
	unsigned long long *outvalue, int absolute)
{
	int cnt;
	thread_t *Sthread;
	int set_id = HardwareCounters_GetCurrentSet(ptask, task, thread);

	Sthread = GET_THREAD_INFO(ptask, task, thread);

	/* Don't emit hwc that coincide in time  with a hardware counter group change.
	   Special treatment for the first HWC change, which must be excluded in order
	   to get the first counters (which shall be 0).
	   However, we must track the value of counters if SAMPLING_SUPPORT */
	if (Sthread->last_hw_group_change == time && Sthread->HWCChange_count == 1)
	{
#if defined(PAPI_COUNTERS) && defined (SAMPLING_SUPPORT)
		for (cnt = 0; cnt < MAX_HWC; cnt++)
			if (Sthread->HWCSets[set_id][cnt] != NO_COUNTER &&
			    Sthread->HWCSets[Sthread->current_HWCSet][cnt] != SAMPLE_COUNTER)
			{
				if (!absolute)
				{
					Sthread->counters[cnt] = 0; /* Event->HWCValues[cnt]; */
					outvalue[cnt] = 0;
					outtype[cnt] = Sthread->HWCSets_types[set_id][cnt];
				}
				else
				{
					Sthread->counters[cnt] = 0; /* Event->HWCValues[cnt]; */
					outvalue[cnt] = 0;
					outtype[cnt] = Sthread->HWCSets_types[set_id][cnt]
					  + HWC_DELTA_ABSOLUTE;
				}
			}
			else
				outtype[cnt] = NO_COUNTER;
#endif
		return TRUE;
	}
	else if (Sthread->last_hw_group_change == time && Sthread->HWCChange_count > 1)
	{
#if defined(PAPI_COUNTERS) && defined (SAMPLING_SUPPORT)
		for (cnt = 0; cnt < MAX_HWC; cnt++)
			if (Sthread->HWCSets[set_id][cnt] != NO_COUNTER &&
			    Sthread->HWCSets[Sthread->current_HWCSet][cnt] != SAMPLE_COUNTER)
				Sthread->counters[cnt] = Event->HWCValues[cnt];
#endif
		return TRUE;
	}

	for (cnt = 0; cnt < MAX_HWC; cnt++)
	{
		/* If using PAPI, they can be stored in absolute or relative manner,
		   depending whether sampling was activated or not */
#if defined(PAPI_COUNTERS)
# if defined(SAMPLING_SUPPORT)
		if (Sthread->HWCSets[set_id][cnt] != NO_COUNTER &&
		    Sthread->HWCSets[Sthread->current_HWCSet][cnt] != SAMPLE_COUNTER)
# else
		if (Sthread->HWCSets[set_id][cnt] != NO_COUNTER)
# endif
		{
			/* If sampling is enabled PAPI_reset is not called and we must substract
			   the previous read value from the current PAPI_read because it's always
			   adding */

# if defined(SAMPLING_SUPPORT)
			/* Protect when counters are incorrect (major timestamp, lower counter value) */
			if (Event->HWCValues[cnt] >= Sthread->counters[cnt])
			{
				if (!absolute)
				{
					outvalue[cnt] = Event->HWCValues[cnt] - Sthread->counters[cnt];
					outtype[cnt] = Sthread->HWCSets_types[set_id][cnt];
				}
				else
				{
					outvalue[cnt] = Event->HWCValues[cnt];
					outtype[cnt] = Sthread->HWCSets_types[set_id][cnt]
					  + HWC_DELTA_ABSOLUTE;
				}
			}
			else
			{
				outtype[cnt] = NO_COUNTER;
			}
# else
			if (!absolute)
			{
				outvalue[cnt] = Event->HWCValues[cnt];
				outtype[cnt] = Sthread->HWCSets_types[set_id][cnt];
			}
			else
			{
				outvalue[cnt] = Event->HWCValues[cnt];
				outtype[cnt] = Sthread->HWCSets_types[set_id][cnt]
				  + HWC_DELTA_ABSOLUTE;
			}
# endif

			Sthread->counters[cnt] = Event->HWCValues[cnt];
		}
		else
			outtype[cnt] = NO_COUNTER;

#elif defined(PMAPI_COUNTERS)

		if (Sthread->HWCSets[set_id][cnt] != NO_COUNTER)
		{
			if (!absolute)
			{
				outvalue[cnt]= Event->HWCValues[cnt] - Sthread->counters[cnt];
				outtype[cnt] = HWC_COUNTER_TYPE (cnt, Sthread->HWCSets[set_id][cnt]);
				Sthread->counters[cnt] = Event->HWCValues[cnt];
			}
			else
			{
				outvalue[cnt]= Event->HWCValues[cnt] - Sthread->counters[cnt];
				outtype[cnt] = HWC_COUNTER_TYPE (cnt, Sthread->HWCSets[set_id][cnt]) 
				  + HWC_DELTA_ABSOLUTE;
			}
		}
		else
			outtype[cnt] = NO_COUNTER;

#endif

	}
	return TRUE;
}

static int HardwareCounters_Compare (const int *HWC1, const int *used1,
	const int *HWC2, const int *used2)
{
	int i;

	for (i = 0; i < MAX_HWC; i++)
		if ((HWC1[i] != HWC2[i]) || (used1[i] != used2[i]))
			return FALSE;

	return TRUE;
}

static int HardwareCounters_Exist (const int *HWC, const int *used)
{
	CntQueue *queue, *ptmp;

	queue = &CountersTraced;
	for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev)
		if (HardwareCounters_Compare (ptmp->Events, ptmp->Traced, HWC, used))
			return TRUE;

	return FALSE;
}

void HardwareCounters_Show (const event_t * Event, int ncounters)
{
  int cnt;
  fprintf (stdout, "COUNTERS: ");
  for (cnt = 0; cnt < ncounters; cnt++)
    fprintf (stdout, "[%lld] ", Event->HWCValues[cnt]);
  fprintf (stdout, "\n");
}

void HardwareCounters_Get (const event_t *Event, unsigned long long *buffer)
{
	int cnt;
	for (cnt = 0; cnt < MAX_HWC; cnt++)
		buffer[cnt] = Event->HWCValues[cnt];
}

void HardwareCounters_NewSetDefinition (int ptask, int task, int thread, int newSet, long long *HWCIds)
{
	thread_t *Sthread;

	Sthread = GET_THREAD_INFO(ptask, task, thread);

	if (newSet <= Sthread->num_HWCSets)
	{
		int i, j;

		xrealloc(Sthread->HWCSets, Sthread->HWCSets, (newSet+1)*sizeof(int *));
		xmalloc(Sthread->HWCSets[newSet], MAX_HWC*sizeof(int));
		xrealloc(Sthread->HWCSets_types, Sthread->HWCSets_types, (newSet+1)*sizeof(int *));
		xmalloc(Sthread->HWCSets_types[newSet], MAX_HWC*sizeof(int));

		for (i=Sthread->num_HWCSets; i<newSet; i++)
		{
		/* New set definitions should appear ordered. If there's any gap, 
		 * this initializes the missing definition with NO_COUNTER's 
		 */
			for (j=0; j<MAX_HWC; j++)
			{
				Sthread->HWCSets[i][j] = NO_COUNTER;
			}
		}

		for (i=0; i<MAX_HWC; i++)
		{
			if (HWCIds != NULL)
			{
				Sthread->HWCSets[newSet][i] = (int)HWCIds[i];
				Sthread->HWCSets_types[newSet][i] = HWC_COUNTER_TYPE(HWCIds[i]);
			}
			else
				Sthread->HWCSets[newSet][i] = NO_COUNTER;
		}
		Sthread->num_HWCSets = newSet + 1;
	}
}

int * HardwareCounters_GetSetIds(int ptask, int task, int thread, int set_id)
{
	thread_t *Sthread;
	static int warn_count = 0;

	Sthread = GET_THREAD_INFO(ptask, task, thread);

	if ((set_id+1 > Sthread->num_HWCSets) || (set_id < 0))
	{
		warn_count ++;
		if (warn_count < 10)
		{
			fprintf(stderr, "\nmpi2prv: WARNING! Definitions for HWC set '%d' were not found for object (%d.%d.%d)!\n"
			"You're probably using an old version of the tracing library, please upgrade it!\n", set_id, ptask, task, thread);
		}
		else if (warn_count == 10)
		{
			fprintf(stderr, "(Future warnings will be omitted...)\n");
		}
		HardwareCounters_NewSetDefinition(ptask, task, thread, set_id, NULL);
	}
 	return Sthread->HWCSets[set_id];
}

int HardwareCounters_GetCurrentSet(int ptask, int task, int thread)
{
	thread_t *Sthread;
	Sthread = GET_THREAD_INFO(ptask, task, thread);

	return Sthread->current_HWCSet;
}

/******************************************************************************
 **      Function name : HardwareCounters_Change
 **      
 **      Description : 
 ******************************************************************************/

void HardwareCounters_Change (int ptask, int task, int thread,
	int newSet, int *outtypes, unsigned long long *outvalues)
{
	int cnt;
	CntQueue *cItem;
	thread_t *Sthread;
	int counters_used[MAX_HWC];

	int *newIds = HardwareCounters_GetSetIds (ptask, task, thread, newSet);
	Sthread = GET_THREAD_INFO(ptask, task, thread);

	for (cnt = 0; cnt < MAX_HWC; cnt++)
	{
		counters_used[cnt] = (newIds[cnt] != NO_COUNTER);
	}

	outtypes[0] = HWC_GROUP_ID; outvalues[0] = 1+newSet;

	Sthread->current_HWCSet = newSet;
	for (cnt = 0; cnt < MAX_HWC; cnt++)
	{
		Sthread->counters[cnt] = 0;

		/* Emit counters with value 0 at the very beginning*/
		if (counters_used[cnt])
		{
#if defined(PMAPI_COUNTERS)
			outtypes[cnt+1] = HWC_COUNTER_TYPE(cnt, Sthread->HWCSets[newSet][cnt]);
#elif defined(PAPI_COUNTERS)
			outtypes[cnt+1] = Sthread->HWCSets_types[newSet][cnt];
#endif
			outvalues[cnt+1] = 0;
		}
		else
		{
			outtypes[cnt+1] = NO_COUNTER;
		}
	}

	/* Add this counters (if didn't exist) to a queue in order to put them into the PCF */
	if (HardwareCounters_Exist (newIds, counters_used))
	{
		return;
	}

	ALLOC_NEW_ITEM (FreeListItems, sizeof (CntQueue), cItem, "CntQueue");
	for (cnt = 0; cnt < MAX_HWC; cnt++)
	{
		cItem->Events[cnt] = newIds[cnt];
		cItem->Traced[cnt] = (newIds[cnt] != NO_COUNTER);
	}
	ENQUEUE_ITEM (&CountersTraced, cItem);
}

void HardwareCounters_SetOverflow (int ptask, int task, int thread, event_t *Event)
{
	int cnt;
	thread_t *Sthread;
	int set_id = HardwareCounters_GetCurrentSet(ptask, task, thread);

	Sthread = GET_THREAD_INFO(ptask, task, thread);

	for (cnt = 0; cnt < MAX_HWC; cnt++)
		if (Event->HWCValues[cnt] == SAMPLE_COUNTER)
			Sthread->HWCSets[set_id][cnt] = SAMPLE_COUNTER;
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-tags.h"
#include "mpi-aux.h"

static void HardwareCounters_Add (int *HWCValues, int *used)
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
		int counters[MAX_HWC];
		int used[MAX_HWC];

		for (slave = 1; slave < size; slave++)
		{
			/* How many set of counters has each slave? */
			res = MPI_Recv (&ncounters, 1, MPI_INT, slave, NUMBER_OF_HWC_SETS_TAG, MPI_COMM_WORLD, &s);
			MPI_CHECK(res, MPI_Recv, "Receiving number of sets of HWC");

			res = MPI_Send (&ncounters, 1, MPI_INT, slave, HWC_SETS_READY, MPI_COMM_WORLD);
			MPI_CHECK(res, MPI_Send, "Sending ready statement");

			if (ncounters > 0)
			{
				int i;
				/* Just receive the counters of each slave */
				for (i = 0; i < ncounters; i++)
				{
					res = MPI_Recv (counters, MAX_HWC, MPI_INT, slave, HWC_SETS_TAG, MPI_COMM_WORLD, &s);
					MPI_CHECK(res, MPI_Recv, "Receiving HWC");
					res = MPI_Recv (used, MAX_HWC, MPI_INT, slave, HWC_SETS_ENABLED_TAG, MPI_COMM_WORLD, &s);
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

		res = MPI_Send (&count, 1, MPI_INT, 0, NUMBER_OF_HWC_SETS_TAG, MPI_COMM_WORLD);
		MPI_CHECK(res, MPI_Send, "Sending number of HWC sets");

		res = MPI_Recv (&count, 1, MPI_INT, 0, HWC_SETS_READY, MPI_COMM_WORLD, &s);
		MPI_CHECK(res, MPI_Recv, "Receiving ready statement");

		if (count > 0)
		{
  		queue = &CountersTraced;
 		 	for (ptmp = (queue)->prev; ptmp != (queue); ptmp = ptmp->prev)
			{
				res = MPI_Send (ptmp->Events, MAX_HWC, MPI_INT, 0, HWC_SETS_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Sending HWC");
				res = MPI_Send (ptmp->Traced, MAX_HWC, MPI_INT, 0, HWC_SETS_ENABLED_TAG, MPI_COMM_WORLD);
				MPI_CHECK(res, MPI_Send, "Sending used HWC bitmap");
			}
		}
	}
}
#endif

#endif /* USE_HARDWARE_COUNTERS  || HETEROGENEOUS_SUPPORT*/


