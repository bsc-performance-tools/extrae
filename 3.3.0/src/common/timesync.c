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

#include <config.h>

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif

#include "common.h"
#include "types.h"
#include "timesync.h"
#include "utils.h"


/* Time synchronization module */

static INT64 **LatencyTable = NULL;
static SyncInfo_t **SyncInfo = NULL;
static int TotalAppsToSync;
static int *TotalTasksToSync;

static char **NodeList = NULL;
static int TotalNodes = 0;

static int TimeSync_Initialized = FALSE;

/**
 * Translates the name of a node into a numeric identifier
 * @param[in] node The node name
 * @return A numeric identifier for the given node
 */
static int Get_NodeId (char *node)
{
	int i;

	for (i=0; i<TotalNodes; i++)
		if (!strcmp(node, NodeList[i]))
		{
			/* Found */
			return i;
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
int TimeSync_Initialize (int num_appls, int *num_tasks)
{
	int i, j;

#if defined(DEBUG)
	fprintf (stderr, "DEBUG: TimeSync_Initialize (%d, %p)\n", num_appls, num_tasks);
#endif

	ASSERT(num_appls>0, "Invalid number of applications in TimeSync_Initialize");
	ASSERT(num_tasks!=NULL, "Invalid set of tasks in TimeSync_Initialize");

	TotalAppsToSync = num_appls;
	TotalTasksToSync = (int*) malloc (sizeof(int)*num_appls);
	ASSERT(TotalTasksToSync!=NULL, "Cannot allocate memory to synchronize application tasks");
	for (i = 0; i < num_appls; i++)
	{
#if defined(DEBUG)
		fprintf (stderr, "DEBUG: TotalTasksToSync[%d] = %d\n", i, num_tasks[i]);
#endif
		TotalTasksToSync[i] = num_tasks[i];
	}

	LatencyTable = (INT64**) malloc (sizeof(INT64*)*num_appls);
	ASSERT(LatencyTable!=NULL, "Cannot allocate latency table to synchronize application tasks");
	for (i = 0; i < num_appls; i++)
	{
		LatencyTable[i] = (INT64*) malloc (sizeof(INT64)*num_tasks[i]);
		ASSERT(LatencyTable[i]!=NULL, "Cannot allocate latency table to synchronize application task");
	}
	SyncInfo = (SyncInfo_t **) malloc (sizeof(SyncInfo_t*)*num_appls);
	ASSERT(SyncInfo!=NULL, "Cannot allocate synchronization table to synchronize application tasks");
	for (i = 0; i < num_appls; i++)
	{
		SyncInfo[i] = (SyncInfo_t*) malloc (sizeof(SyncInfo_t)*num_tasks[i]);
		ASSERT(SyncInfo[i]!=NULL, "Cannot allocate synchronization table to synchronize application task");
	}

	for (i = 0; i < num_appls; i++)
		for (j = 0; j < num_tasks[i]; j++)
		{
			LatencyTable[i][j] = 0;
			SyncInfo[i][j].init = FALSE;
			SyncInfo[i][j].init_time = 0;
			SyncInfo[i][j].sync_time = 0;
			SyncInfo[i][j].node_id = 0;
		}

	TimeSync_Initialized = TRUE;

	return 1;
}

/**
 * Frees the allocated structures
 */
void TimeSync_CleanUp (void)
{
	int i;

#if defined(DEBUG)
	fprintf (stderr, "DEBUG: TimeSync_CleanUp\n");
#endif

	for (i = 0; i < TotalAppsToSync; i++)
	{	
		xfree (SyncInfo[i]);
		xfree (LatencyTable[i]);
	}
	xfree (SyncInfo);
	xfree (LatencyTable);

	for (i = 0; i < TotalNodes; i++)
		xfree (NodeList[i]);
	xfree (NodeList);
	TotalNodes = 0;

	TotalAppsToSync = 0;
	xfree (TotalTasksToSync);
}


/**
 * Sets the synchronization times of each task
 * @param[in] task The task identifier
 * @param[in] init_time Time of the first event of this task
 * @param[in] sync_time Time of the synchronization point of this task
 * @param[in] node Name of the node where this task is executing
 * @return 1 upon successful setup, 0 otherwise
 */
int TimeSync_SetInitialTime (int app, int task, UINT64 init_time, UINT64 sync_time, char *node)
{
#if defined(DEBUG)
	fprintf(stderr, "DEBUG: [TS %d.%d] TimeSync_SetInitialTime %llu %llu %s\n", app, task, init_time, sync_time, node);
#endif

	ASSERT(TimeSync_Initialized && app >= 0 && app < TotalAppsToSync && task >= 0 && task < TotalTasksToSync[app],
	  "TimeSync module was not correctly initialized!");

	SyncInfo[app][task].init = TRUE;
	SyncInfo[app][task].init_time = init_time;
	SyncInfo[app][task].sync_time = sync_time;
	SyncInfo[app][task].node_id = Get_NodeId(node);
	return 1;
}

/**
 * Calculates the latencies for each task depending on the specified strategy
 * @param[in] sync_strategy Choose between per task, per node or no synchronization.
 * @return 1 upon success, 0 otherwise
 */
int TimeSync_CalculateLatencies (int sync_strategy)
{
	int i, j;
	UINT64 min_init_time = 0, max_sync_time = 0;

#if defined(DEBUG)
	fprintf(stderr, "DEBUG: TimeSync_CalculateLatencies (%d)\n", sync_strategy);
#endif

	/* Check all tasks are initialized */
	for (i=0; i<TotalAppsToSync; i++)
		for (j = 0; j < TotalTasksToSync[i]; j++)
			if (!SyncInfo[i][j].init)
			{
				fprintf(stderr, "WARNING: TimeSync_CalculateLatencies: Task %i was not initialized. Synchronization disabled!\n", i);
				return 0;
			}

	if (sync_strategy == TS_TASK)
	{
		/* Calculate the maximum synchronization time */
		for (i = 0; i < TotalAppsToSync; i++)
			for (j = 0; j < TotalTasksToSync[i]; j++)
				max_sync_time = MAX(max_sync_time, SyncInfo[i][j].sync_time);

		/* Move the other tasks to the right to match the synchronization points */
		for (i = 0; i<TotalAppsToSync; i++)
			for (j = 0; j < TotalTasksToSync[i]; j++)
				LatencyTable[i][j] = max_sync_time - SyncInfo[i][j].sync_time;
	}
	else if ((sync_strategy == TS_NODE) || (sync_strategy == TS_DEFAULT))
	{
		UINT64 *max_sync_time_per_node;

		max_sync_time_per_node = (UINT64 *)malloc(sizeof(UINT64) * TotalNodes);
		memset(max_sync_time_per_node, 0,sizeof(UINT64) * TotalNodes);

		/* Calculate the maximum synchronization time per node */
		for (i=0; i<TotalAppsToSync; i++)
			for (j = 0; j < TotalTasksToSync[i]; j++)
				max_sync_time_per_node[SyncInfo[i][j].node_id] = MAX(max_sync_time_per_node[SyncInfo[i][j].node_id], SyncInfo[i][j].sync_time);

		/* Calculate the absolute maximum synchronization time */
		for (i=0; i<TotalNodes; i++)
			max_sync_time = MAX(max_sync_time, max_sync_time_per_node[i]);

		/* Move all tasks of the other nodes to the right */
		for (i=0; i<TotalAppsToSync; i++)
			for (j = 0; j < TotalTasksToSync[i]; j++)
				LatencyTable[i][j] = max_sync_time - max_sync_time_per_node[SyncInfo[i][j].node_id];

		free (max_sync_time_per_node);
	}

	/* Calculate the minimum first time (latencies are already applied) */
	min_init_time = SyncInfo[0][0].init_time + LatencyTable[0][0];
	for (i = 0; i < TotalAppsToSync; i++)
		for (j = 0; j < TotalTasksToSync[i]; j++)
			min_init_time = MIN(min_init_time, SyncInfo[i][j].init_time + LatencyTable[i][j]);

	/* Move tasks to the left to make them start at 0 */
	for (i = 0; i<TotalAppsToSync; i++)
		for (j = 0; j < TotalTasksToSync[i]; j++)
			LatencyTable[i][j] = (INT64)(LatencyTable[i][j] - min_init_time);

	/* fprintf(stderr, "LatencyTable[%d] = %lld\n", i, LatencyTable[i]); */

	return 1;
}

/**
 * Synchronizes the given time
 * @param[in] task The task to synchronize
 * @param[in] time The time to synchronize
 * @return The synchronized time
 */
UINT64 TimeSync (int app, int task, UINT64 time)
{
	return (UINT64)((INT64)time + LatencyTable[app][task]);
}

/**
 * Returns a synchronized time to its original value
 * @param[in] task The task to desynchronize
 * @param[in] time The time to desynchronize
 * @return The desynchronized time
 */
UINT64 TimeDesync (int app, int task, UINT64 time)
{   
    return (UINT64)((INT64)time - LatencyTable[app][task]);
}

#if defined(BUILD_EXECUTABLE)
int main(int argc, char **argv)
{
	int ntasks = 4;
	TimeSync_Initialize (1, &ntasks);
	TimeSync_SetInitialTime (0, 0, 20, 80, "node1");
	TimeSync_SetInitialTime (0, 1, 10, 30, "node1");
	TimeSync_SetInitialTime (0, 2, 5,  75, "node2");
	TimeSync_SetInitialTime (0, 3, 15, 60, "node2");
	TimeSync_CalculateLatencies (TS_NODE);
	fprintf(stderr, "TIMESYNC(0, 0, 20) = %llu\n", TIMESYNC(0, 0, 20));	
	fprintf(stderr, "TIMESYNC(0, 0, 80) = %llu\n", TIMESYNC(0, 0, 80));	
	fprintf(stderr, "TIMESYNC(0, 1, 10) = %llu\n", TIMESYNC(0, 1, 10));	
	fprintf(stderr, "TIMESYNC(0, 1, 30) = %llu\n", TIMESYNC(0, 1, 30));	
	fprintf(stderr, "TIMESYNC(0, 2, 5)  = %llu\n", TIMESYNC(0, 2, 5));	
	fprintf(stderr, "TIMESYNC(0, 2, 75) = %llu\n", TIMESYNC(0, 2, 75));	
	fprintf(stderr, "TIMESYNC(0, 3, 15) = %llu\n", TIMESYNC(0, 3, 15));	
	fprintf(stderr, "TIMESYNC(0, 3, 60) = %llu\n", TIMESYNC(0, 3, 60));	

	return 0;
}
#endif
