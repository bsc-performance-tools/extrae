/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include "common.h"
#include "types.h"
#include "timesync.h"

/* Time synchronization module */

INT64 *LatencyTable = NULL;
SyncInfo_t *SyncInfo = NULL;
int TotalTasksToSync = 0;

char **NodeList = NULL;
int TotalNodes = 0;

int TimeSync_Initialized = FALSE;

/**
 * Translates the name of a node into a numeric identifier
 * @param[in] node The node name
 * @return A numeric identifier for the given node
 */
static int Get_NodeId (char *node)
{
	int i;

	for (i=0; i<TotalNodes; i++)
	{
		if (!strcmp(node, NodeList[i]))
		{
			/* Found */
			return i;
		}
	}
	/* Not found */
	TotalNodes ++;
	NodeList = (char **)realloc(NodeList, TotalNodes * sizeof(char *));
	NodeList[TotalNodes - 1] = (char *)malloc(strlen(node) + 1);
	strcpy (NodeList[TotalNodes - 1], node);

	return TotalNodes - 1;
}

/**
 * Initializes the time synchronization module 
 * @param[in] num_tasks Total number of tasks 
 * @return 1 upon successful initialization, 0 otherwise
 */
int TimeSync_Initialize (int num_tasks)
{
	int i;

	if (num_tasks > 0)
	{
		TotalTasksToSync = num_tasks;
	
		LatencyTable = (INT64 *)malloc(TotalTasksToSync * sizeof(INT64));
		if (LatencyTable == NULL)
		{
			fprintf(stderr, "Error: TimeSync module: Not enough memory to allocate LatencyTable\n");
			perror("malloc");
			exit(1);
		}
		SyncInfo = (SyncInfo_t *)malloc(TotalTasksToSync * sizeof(SyncInfo_t));
		if (SyncInfo == NULL)
		{
			fprintf(stderr, "Error: TimeSync module: Not enough memory to allocate SyncInfo\n");
			perror("malloc");
			exit(1);
		}
		for (i=0; i<TotalTasksToSync; i++)
		{
			LatencyTable[i] = 0;
	
			SyncInfo[i].init = FALSE;
			SyncInfo[i].init_time = 0;
			SyncInfo[i].sync_time = 0;
			SyncInfo[i].node_id = 0;
		}

		TimeSync_Initialized = TRUE;

		return 1;	
	}
	return 0;
}

/**
 * Sets the synchronization times of each task
 * @param[in] task The task identifier
 * @param[in] init_time Time of the first event of this task
 * @param[in] sync_time Time of the synchronization point of this task
 * @param[in] node Name of the node where this task is executing
 * @return 1 upon successful setup, 0 otherwise
 */
int TimeSync_SetInitialTime (int task, UINT64 init_time, UINT64 sync_time, char *node)
{
	/* fprintf(stderr, "[TS %d] TimeSync_SetInitialTime %llu %llu %s\n", task, init_time, sync_time, node); */
	if ((TimeSync_Initialized) && (task >= 0) && (task < TotalTasksToSync))
	{
		SyncInfo[task].init = TRUE;
		SyncInfo[task].init_time = init_time;
		SyncInfo[task].sync_time = sync_time;
		SyncInfo[task].node_id = Get_NodeId(node);
		return 1;
	}
	else
	{
		fprintf(stderr, "WARNING: TimeSync module not correctly initialized (TotalTasks=%d, CurrentTask=%d)\n", TotalTasksToSync, task);
		return 0;
	}
}

/**
 * Calculates the latencies for each task depending on the specified strategy
 * @param[in] sync_strategy Choose between per task, per node or no synchronization.
 * @return 1 upon success, 0 otherwise
 */
int TimeSync_CalculateLatencies (int sync_strategy)
{
	int i;
	UINT64 min_init_time = 0, max_sync_time = 0;

	/* Check all tasks are initialized */
	for (i=0; i<TotalTasksToSync; i++)
	{
		if (!SyncInfo[i].init) {
			fprintf(stderr, "WARNING: TimeSync_CalculateLatencies: Task %i was not initialized. Synchronization disabled!\n", i);
			return 0;
		}
	}

	if (sync_strategy == TS_TASK)
	{
		/* Calculate the maximum synchronization time */
		for (i=0; i<TotalTasksToSync; i++)
		{
			max_sync_time = MAX(max_sync_time, SyncInfo[i].sync_time);
		}
		/* Move the other tasks to the right to match the synchronization points */
		for (i=0; i<TotalTasksToSync; i++)
		{
			LatencyTable[i] = max_sync_time - SyncInfo[i].sync_time;
		}
	}
	else if ((sync_strategy == TS_NODE) || (sync_strategy == TS_DEFAULT))
	{
		UINT64 *max_sync_time_per_node;

		max_sync_time_per_node = (UINT64 *)malloc(sizeof(UINT64) * TotalNodes);
		bzero(max_sync_time_per_node, sizeof(UINT64) * TotalNodes);

		/* Calculate the maximum synchronization time per node */
		for (i=0; i<TotalTasksToSync; i++)
		{
			max_sync_time_per_node[SyncInfo[i].node_id] = MAX(max_sync_time_per_node[SyncInfo[i].node_id], SyncInfo[i].sync_time);
		}
		/* Calculate the absolute maximum synchronization time */
		for (i=0; i<TotalNodes; i++)
		{
			max_sync_time = MAX(max_sync_time, max_sync_time_per_node[i]);
		}
		/* Move all tasks of the other nodes to the right */
		for (i=0; i<TotalTasksToSync; i++)
		{
			LatencyTable[i] = max_sync_time - max_sync_time_per_node[SyncInfo[i].node_id];
		}
		free (max_sync_time_per_node);
	}

	/* Calculate the minimum first time (latencies are already applied) */
	min_init_time = SyncInfo[0].init_time + LatencyTable[0];
	for (i=0; i<TotalTasksToSync; i++)
	{
		min_init_time = MIN(min_init_time, SyncInfo[i].init_time + LatencyTable[i]);
	}

	/* Move tasks to the left to make them start at 0 */
	for (i=0; i<TotalTasksToSync; i++)
	{
		LatencyTable[i] = (INT64)(LatencyTable[i] - min_init_time);
		/* fprintf(stderr, "LatencyTable[%d] = %lld\n", i, LatencyTable[i]); */
	}

	return 1;
}

/**
 * Synchronizes the given time
 * @param[in] task The task to synchronize
 * @param[in] time The time to synchronize
 * @return The synchronized time
 */
UINT64 TimeSync (int task, UINT64 time)
{
	return (UINT64)((INT64)time + LatencyTable[task]);
}

/**
 * Returns a synchronized time to its original value
 * @param[in] task The task to desynchronize
 * @param[in] time The time to desynchronize
 * @return The desynchronized time
 */
UINT64 TimeDesync (int task, UINT64 time)
{   
    return (UINT64)((INT64)time - LatencyTable[task]);
}

#if defined(BUILD_EXECUTABLE)
int main(int argc, char **argv)
{
	TimeSync_Initialize (4);
	TimeSync_SetInitialTime (0, 20, 80, "node1");
	TimeSync_SetInitialTime (1, 10, 30, "node1");
	TimeSync_SetInitialTime (2, 5,  75, "node2");
	TimeSync_SetInitialTime (3, 15, 60, "node2");
	TimeSync_CalculateLatencies (TS_NODE);
	fprintf(stderr, "TIMESYNC(0, 20) = %llu\n", TIMESYNC(0, 20));	
	fprintf(stderr, "TIMESYNC(0, 80) = %llu\n", TIMESYNC(0, 80));	
	fprintf(stderr, "TIMESYNC(1, 10) = %llu\n", TIMESYNC(1, 10));	
	fprintf(stderr, "TIMESYNC(1, 30) = %llu\n", TIMESYNC(1, 30));	
	fprintf(stderr, "TIMESYNC(2, 5)  = %llu\n", TIMESYNC(2, 5));	
	fprintf(stderr, "TIMESYNC(2, 75) = %llu\n", TIMESYNC(2, 75));	
	fprintf(stderr, "TIMESYNC(3, 15) = %llu\n", TIMESYNC(3, 15));	
	fprintf(stderr, "TIMESYNC(3, 60) = %llu\n", TIMESYNC(3, 60));	

	return 0;
}
#endif
