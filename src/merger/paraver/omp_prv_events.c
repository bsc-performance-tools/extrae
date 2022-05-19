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

#ifdef HAVE_BFD
# include "addr2info.h"
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "events.h"
#include "omp_prv_events.h"
#include "mpi2out.h"
#include "options.h"

/* New OMP support */
enum
{
  NEW_OMP_CALL_INDEX,
  NEW_OMP_NESTED_INDEX,
  NEW_OMP_PARALLEL_INDEX,
  NEW_OMP_WORKSHARING_INDEX,
  NEW_OMP_SYNC_INDEX,
  NEW_OMP_LOCK_INDEX,
  NEW_OMP_LOCK_NAME_INDEX,
  NEW_OMP_ORDERED_INDEX,
  NEW_OMP_TASKGROUP_INDEX,
  NEW_OMP_OUTLINED_INDEX,
  NEW_OMP_TASKING_INDEX,
  NEW_OMP_TASK_INST_INDEX,
  NEW_OMP_TASK_EXEC_INDEX,
  NEW_MAX_OMP_INDEX
};

static int new_inuse[NEW_MAX_OMP_INDEX] = { FALSE };

void NEW_Enable_OMP_Operation (int type)
{
  if (type == NEW_OMP_CALL_EV)
    new_inuse[NEW_OMP_CALL_INDEX] = TRUE;
  else if (type == NEW_OMP_NESTED_EV)
    new_inuse[NEW_OMP_NESTED_INDEX] = TRUE;
  else if (type == NEW_OMP_PARALLEL_EV)
    new_inuse[NEW_OMP_PARALLEL_INDEX] = TRUE;
  else if (type == NEW_OMP_WSH_EV)
    new_inuse[NEW_OMP_WORKSHARING_INDEX] = TRUE;
  else if (type == NEW_OMP_SYNC_EV)
    new_inuse[NEW_OMP_SYNC_INDEX] = TRUE;
  else if (type == NEW_OMP_LOCK_EV)
    new_inuse[NEW_OMP_LOCK_INDEX] = TRUE;
  else if (type == NEW_OMP_LOCK_NAME_EV)
    new_inuse[NEW_OMP_LOCK_NAME_INDEX] = TRUE;
  else if (type == NEW_OMP_ORDERED_EV)
    new_inuse[NEW_OMP_ORDERED_INDEX] = TRUE;
  else if (type == NEW_OMP_TASKGROUP_EV)
    new_inuse[NEW_OMP_TASKGROUP_INDEX] = TRUE;
  else if (type == NEW_OMP_OUTLINED_ADDRESS_EV)
    new_inuse[NEW_OMP_OUTLINED_INDEX] = TRUE;
  else if (type == NEW_OMP_TASKING_EV)
    new_inuse[NEW_OMP_TASKING_INDEX] = TRUE;
  else if ((type == NEW_OMP_TASK_INST_ID_EV) || (type == NEW_OMP_TASK_INST_ADDRESS_EV))
    new_inuse[NEW_OMP_TASK_INST_INDEX] = TRUE;
  else if ((type == NEW_OMP_TASK_EXEC_ID_EV) || (type == NEW_OMP_TASK_EXEC_ADDRESS_EV))
    new_inuse[NEW_OMP_TASK_EXEC_INDEX] = TRUE;
}

char *OMP_Call_Name[MAX_OMP_CALLS] =
{
  [ 0 ] = "Exiting",
  [ GOMP_ATOMIC_START_VAL ] = "GOMP_atomic_start",
  [ GOMP_ATOMIC_END_VAL ] = "GOMP_atomic_end",
  [ GOMP_BARRIER_VAL ] = "GOMP_barrier",
  [ GOMP_CRITICAL_START_VAL ] = "GOMP_critical_start",
  [ GOMP_CRITICAL_END_VAL ] = "GOMP_critical_end",
  [ GOMP_CRITICAL_NAME_START_VAL ] = "GOMP_critical_name_start",
  [ GOMP_CRITICAL_NAME_END_VAL ] = "GOMP_critical_name_end",
  [ GOMP_LOOP_STATIC_START_VAL ] = "GOMP_loop_static_start",
  [ GOMP_LOOP_DYNAMIC_START_VAL ] = "GOMP_loop_dynamic_start",
  [ GOMP_LOOP_GUIDED_START_VAL ] = "GOMP_loop_guided_start",
  [ GOMP_LOOP_RUNTIME_START_VAL ] = "GOMP_loop_runtime_start",
  [ GOMP_LOOP_STATIC_NEXT_VAL ] = "GOMP_loop_static_next",
  [ GOMP_LOOP_DYNAMIC_NEXT_VAL ] = "GOMP_loop_dynamic_next",
  [ GOMP_LOOP_GUIDED_NEXT_VAL ] = "GOMP_loop_guided_next",
  [ GOMP_LOOP_RUNTIME_NEXT_VAL ] = "GOMP_loop_runtime_next",
  [ GOMP_LOOP_ORDERED_STATIC_START_VAL ] = "GOMP_loop_ordered_static_start",
  [ GOMP_LOOP_ORDERED_DYNAMIC_START_VAL ] = "GOMP_loop_ordered_dynamic_start",
  [ GOMP_LOOP_ORDERED_GUIDED_START_VAL ] = "GOMP_loop_ordered_guided_start",
  [ GOMP_LOOP_ORDERED_RUNTIME_START_VAL ] = "GOMP_loop_ordered_runtime_start",
  [ GOMP_LOOP_ORDERED_STATIC_NEXT_VAL ] = "GOMP_loop_ordered_static_next",
  [ GOMP_LOOP_ORDERED_DYNAMIC_NEXT_VAL ] = "GOMP_loop_ordered_dynamic_next",
  [ GOMP_LOOP_ORDERED_GUIDED_NEXT_VAL ] = "GOMP_loop_ordered_guided_next",
  [ GOMP_LOOP_ORDERED_RUNTIME_NEXT_VAL ] = "GOMP_loop_ordered_runtime_next",
  [ GOMP_PARALLEL_LOOP_STATIC_START_VAL ] = "GOMP_parallel_loop_static_start",
  [ GOMP_PARALLEL_LOOP_DYNAMIC_START_VAL ] = "GOMP_parallel_loop_dynamic_start",
  [ GOMP_PARALLEL_LOOP_GUIDED_START_VAL ] = "GOMP_parallel_loop_guided_start",
  [ GOMP_PARALLEL_LOOP_RUNTIME_START_VAL ] = "GOMP_parallel_loop_runtime_start",
  [ GOMP_LOOP_END_VAL ] = "GOMP_loop_end",
  [ GOMP_LOOP_END_NOWAIT_VAL ] = "GOMP_loop_end_nowait",
  [ GOMP_ORDERED_START_VAL ] = "GOMP_ordered_start",
  [ GOMP_ORDERED_END_VAL ] = "GOMP_ordered_end",
  [ GOMP_PARALLEL_START_VAL ] = "GOMP_parallel_start",
  [ GOMP_PARALLEL_END_VAL ] = "GOMP_parallel_end",
  [ GOMP_PARALLEL_SECTIONS_START_VAL ] = "GOMP_parallel_sections_start",
  [ GOMP_PARALLEL_SECTIONS_VAL ] = "GOMP_parallel_sections",
  [ GOMP_SECTIONS_START_VAL ] = "GOMP_sections_start",
  [ GOMP_SECTIONS_NEXT_VAL ] = "GOMP_sections_next",
  [ GOMP_SECTIONS_END_VAL ] = "GOMP_sections_end",
  [ GOMP_SECTIONS_END_NOWAIT_VAL ] = "GOMP_sections_end_nowait",
  [ GOMP_SINGLE_START_VAL ] = "GOMP_single_start",
  [ GOMP_TASKWAIT_VAL ] = "GOMP_taskwait",
  [ GOMP_PARALLEL_VAL ] = "GOMP_parallel",
  [ GOMP_PARALLEL_LOOP_STATIC_VAL ] = "GOMP_parallel_loop_static",
  [ GOMP_PARALLEL_LOOP_DYNAMIC_VAL ] = "GOMP_parallel_loop_dynamic",
  [ GOMP_PARALLEL_LOOP_GUIDED_VAL ] = "GOMP_parallel_loop_guided",
  [ GOMP_PARALLEL_LOOP_RUNTIME_VAL ] = "GOMP_parallel_loop_runtime",
  [ GOMP_TASKGROUP_START_VAL ] = "GOMP_taskgroup_start",
  [ GOMP_TASKGROUP_END_VAL ] = "GOMP_taskgroup_end",
  [ GOMP_TASK_VAL ] = "GOMP_task",
  [ GOMP_TASKLOOP_VAL ] = "GOMP_taskloop",
  [ GOMP_LOOP_DOACROSS_STATIC_START_VAL ] = "GOMP_loop_doacross_static_start",
  [ GOMP_LOOP_DOACROSS_DYNAMIC_START_VAL ] = "GOMP_loop_doacross_dynamic_start",
  [ GOMP_LOOP_DOACROSS_GUIDED_START_VAL ] = "GOMP_loop_doacross_guided_start",
  [ GOMP_LOOP_DOACROSS_RUNTIME_START_VAL ] = "GOMP_loop_doacross_runtime_start",
  [ GOMP_DOACROSS_POST_VAL ] = "GOMP_doacross_post",
  [ GOMP_DOACROSS_WAIT_VAL ] = "GOMP_doacross_wait",
  [ GOMP_PARALLEL_LOOP_NONMONOTONIC_DYNAMIC_VAL ] = "GOMP_parallel_loop_nonmonotonic_dynamic",
  [ GOMP_LOOP_NONMONOTONIC_DYNAMIC_START_VAL ] = "GOMP_loop_nonmonotonic_dynamic_start",
  [ GOMP_LOOP_NONMONOTONIC_DYNAMIC_NEXT_VAL ] = "GOMP_loop_nonmonotonic_dynamic_next",
  [ GOMP_PARALLEL_LOOP_NONMONOTONIC_GUIDED_VAL ] = "GOMP_parallel_loop_nonmonotonic_guided",
  [ GOMP_LOOP_NONMONOTONIC_GUIDED_START_VAL ] = "GOMP_loop_nonmonotonic_guided_start",
  [ GOMP_LOOP_NONMONOTONIC_GUIDED_NEXT_VAL ] = "GOMP_loop_nonmonotonic_guided_next",
  [ GOMP_PARALLEL_LOOP_NONMONOTONIC_RUNTIME_VAL ] = "GOMP_parallel_loop_nonmonotonic_runtime",
  [ GOMP_PARALLEL_LOOP_MAYBE_NONMONOTONIC_RUNTIME_VAL ] = "GOMP_parallel_loop_maybe_nonmonotonic_runtime",
  [ GOMP_LOOP_NONMONOTONIC_RUNTIME_START_VAL ] = "GOMP_loop_nonmonotonic_runtime_start",
  [ GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_START_VAL ] = "GOMP_loop_maybe_nonmonotonic_runtime_start",
  [ GOMP_LOOP_NONMONOTONIC_RUNTIME_NEXT_VAL ] = "GOMP_loop_nonmonotonic_runtime_next",
  [ GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_NEXT_VAL ] = "GOMP_loop_maybe_nonmonotonic_runtime_next",
  [ GOMP_TEAMS_REG_VAL ] = "GOMP_teams_reg",
  [ GOMP_SET_LOCK_VAL ] = "GOMP_set_lock",
  [ GOMP_UNSET_LOCK_VAL ] = "GOMP_unset_lock",
};

/* Deprecated OpenMP support */

#define PAR_OMP_INDEX           0  /* PARALLEL constructs */
#define WSH_OMP_INDEX           1  /* WORKSHARING constructs */
#define FNC_OMP_INDEX           2  /* Pointers to routines <@> */
#define ULCK_OMP_INDEX          3  /* Unnamed locks in use! */
#define LCK_OMP_INDEX           4  /* Named locks in use! */
#define WRK_OMP_INDEX           5  /* Work delivery */
#define JOIN_OMP_INDEX          6  /* Joins */
#define BARRIER_OMP_INDEX       7  /* Barriers */
#define GETSETNUMTHREADS_INDEX  8  /* Set or Get num threads */
#define TASK_INDEX              9  /* Task event */
#define TASKWAIT_INDEX          10 /* Taskwait event */
#define OMPT_CRITICAL_INDEX     11
#define OMPT_ATOMIC_INDEX       12
#define OMPT_LOOP_INDEX         13
#define OMPT_WORKSHARE_INDEX    14
#define OMPT_SECTIONS_INDEX     15
#define OMPT_SINGLE_INDEX       16
#define OMPT_MASTER_INDEX       17
#define TASKGROUP_START_INDEX   18
#define OMP_STATS_INDEX         19
#define TASKLOOP_INDEX          20 /* Taskloop event */
#define ORDERED_INDEX           21 /* Ordered section in ordered or doacross loops */

#define MAX_OMP_INDEX           22

static int inuse[MAX_OMP_INDEX] = { FALSE };

/* All extrae routines associated with the previous OMP implementation have been marked
 as OLD or deprecated as a way to indefity them for future deletion when we stop
 the support for this implementation */

void OLD_Enable_OMP_Operation (int type)
{
	if (type == PAR_EV)
		inuse[PAR_OMP_INDEX] = TRUE;
	else if (type == WSH_EV)
		inuse[WSH_OMP_INDEX] = TRUE;
	else if (type == OMPFUNC_EV || type == TASKFUNC_EV || type == OMPT_TASKFUNC_EV)
		inuse[FNC_OMP_INDEX] = TRUE;
	else if (type == UNNAMEDCRIT_EV)
		inuse[ULCK_OMP_INDEX] = TRUE;
	else if (type == NAMEDCRIT_EV)
		inuse[LCK_OMP_INDEX] = TRUE;
	else if (type == WORK_EV)
		inuse[WRK_OMP_INDEX] = TRUE;
	else if (type == JOIN_EV)
		inuse[JOIN_OMP_INDEX] = TRUE;
	else if (type == BARRIEROMP_EV)
		inuse[BARRIER_OMP_INDEX] = TRUE;
	else if (type == OMPGETNUMTHREADS_EV || type == OMPSETNUMTHREADS_EV)
		inuse[GETSETNUMTHREADS_INDEX] = TRUE;
	else if (type == TASK_EV)
		inuse[TASK_INDEX] = TRUE;
	else if (type == TASKWAIT_EV)
		inuse[TASKWAIT_INDEX] = TRUE;
	else if (type == TASKLOOP_EV) 
		inuse[TASKLOOP_INDEX] = TRUE;
	else if (type == ORDERED_EV)
		inuse[ORDERED_INDEX] = TRUE;

#define ENABLE_TYPE_IF(x,type,v) \
	if (x ## _EV == type) \
		v[x ## _INDEX] = TRUE;

	ENABLE_TYPE_IF(OMPT_CRITICAL, type, inuse);
	ENABLE_TYPE_IF(OMPT_ATOMIC, type, inuse);
	ENABLE_TYPE_IF(OMPT_LOOP, type, inuse);
	ENABLE_TYPE_IF(OMPT_WORKSHARE, type, inuse);
	ENABLE_TYPE_IF(OMPT_SECTIONS, type, inuse);
	ENABLE_TYPE_IF(OMPT_SINGLE, type, inuse);
	ENABLE_TYPE_IF(OMPT_MASTER, type, inuse);
	ENABLE_TYPE_IF(TASKGROUP_START, type, inuse);
	if (type == OMPT_TASKGROUP_IN_EV)
		inuse[TASKGROUP_START_INDEX] = TRUE;
	ENABLE_TYPE_IF(OMP_STATS, type, inuse);
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_OMP_Operations (void)
{
	int res, i, tmp[MAX_OMP_INDEX], new_tmp[NEW_MAX_OMP_INDEX];

	/* Deprecated OpenMP support */
	res = MPI_Reduce (inuse, tmp, MAX_OMP_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing OpenMP enabled operations");

	for (i = 0; i < MAX_OMP_INDEX; i++)
		inuse[i] = tmp[i];

	/* New OpenMP support */
	res = MPI_Reduce (new_inuse, new_tmp, NEW_MAX_OMP_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing OpenMP enabled operations");

	for (i = 0; i < NEW_MAX_OMP_INDEX; i++)
	{
		new_inuse[i] = new_tmp[i];
	}
}

#endif

/* Deprecated OpenMP support */
static void OLD_OMPEvent_WriteEnabledOperations (FILE * fd)
{
	if (inuse[JOIN_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d  OpenMP Worksharing join\n", JOIN_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "%d Join (w wait)\n"
		             "%d Join (w/o wait)\n\n",
		             JOIN_WAIT_VAL, JOIN_NOWAIT_VAL);
	}
	if (inuse[WRK_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d  OpenMP Worksharing work dispatcher\n", WORK_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "1 Begin\n\n");
	}
	if (inuse[PAR_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d  Parallel (OMP)\n", PAR_EV);
		fprintf (fd, "VALUES\n"
		             "0 close\n"
		             "1 DO (open)\n"
		             "2 SECTIONS (open)\n"
		             "3 REGION (open)\n\n");
	}
	if (inuse[WSH_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d Worksharing (OMP)\n", WSH_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "4 DO \n"
		             "5 SECTIONS\n"
		             "6 SINGLE\n\n");
	}
#if defined(HAVE_BFD)
	if (inuse[FNC_OMP_INDEX])
	{
		Address2Info_Write_OMP_Labels (fd, OMPFUNC_EV, "Executed OpenMP parallel function",
			OMPFUNC_LINE_EV, "Executed OpenMP parallel function line and file",
			get_option_merge_UniqueCallerID());
		Address2Info_Write_OMP_Labels (fd, TASKFUNC_EV, "Executed OpenMP task function",
			TASKFUNC_LINE_EV, "Executed OpenMP task function line and file",
			get_option_merge_UniqueCallerID());
		Address2Info_Write_OMP_Labels (fd, TASKFUNC_INST_EV, "Instantiated OpenMP task function",
			TASKFUNC_INST_LINE_EV, "Instantiated OpenMP task function line and file",
			get_option_merge_UniqueCallerID());
	}
#endif
	if (inuse[LCK_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OpenMP named-Lock\n", NAMEDCRIT_EV);
		fprintf (fd, "VALUES\n"
		             "%d Unlocked status\n"
		             "%d Lock\n"
		             "%d Unlock\n"
		             "%d Locked status\n\n",
		             UNLOCKED_VAL, LOCK_VAL, UNLOCK_VAL, LOCKED_VAL);

		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OpenMP named-Lock address name\n", NAMEDCRIT_NAME_EV);
	}
	if (inuse[ULCK_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OpenMP unnamed-Lock\n", UNNAMEDCRIT_EV);
		fprintf (fd, "VALUES\n"
		             "%d Unlocked status\n"
		             "%d Lock\n"
		             "%d Unlock\n"
		             "%d Locked status\n\n",
		             UNLOCKED_VAL, LOCK_VAL, UNLOCK_VAL, LOCKED_VAL);
	}
	if (inuse[BARRIER_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OpenMP barrier\n", BARRIEROMP_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n");
	}
	if (inuse[GETSETNUMTHREADS_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OpenMP set num threads\n", OMPSETNUMTHREADS_EV);
		fprintf (fd, "0 %d OpenMP get num threads\n", OMPGETNUMTHREADS_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "1 Begin\n");
	}
	if (inuse[TASKWAIT_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OMP taskwait\n", TASKWAIT_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[TASKLOOP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		  "0 %d Taskloop Identifier\n\n",
		  TASKLOOPID_EV);

		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OMP taskloop\n", TASKLOOP_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[ORDERED_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "0 %d OpenMP ordered section\n", ORDERED_EV);
		fprintf (fd, "VALUES\n"
						     "%d Outside ordered\n"
						     "%d Waiting to enter\n"
						     "%d Signaling the exit\n"
						     "%d Inside ordered\n\n",
						     OUTORDERED_VAL, WAITORDERED_VAL, POSTORDERED_VAL, INORDERED_VAL);
	}
	if (inuse[OMPT_CRITICAL_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "0 %d OMP critical\n"
		             "VALUES\n0 End\n1 Begin\n\n",
		         OMPT_CRITICAL_EV);
	}
	if (inuse[OMPT_ATOMIC_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "0 %d OMP atomic\n"
		             "VALUES\n0 End\n1 Begin\n\n",
		         OMPT_ATOMIC_EV);
	}
	if (inuse[OMPT_LOOP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "0 %d OMP loop\n"
		             "VALUES\n0 End\n1 Begin\n\n",
		         OMPT_LOOP_EV);
	}
	if (inuse[OMPT_WORKSHARE_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "0 %d OMP workshare\n"
		             "VALUES\n0 End\n1 Begin\n\n",
		         OMPT_WORKSHARE_EV);
	}
	if (inuse[OMPT_SECTIONS_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "0 %d OMP sections\n"
		             "VALUES\n0 End\n1 Begin\n\n",
		         OMPT_SECTIONS_EV);
	}
	if (inuse[OMPT_SINGLE_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "0 %d OMP single\n"
		             "VALUES\n0 End\n1 Begin\n\n",
		         OMPT_SINGLE_EV);
	}
	if (inuse[OMPT_MASTER_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "0 %d OMP master\n"
		             "VALUES\n0 End\n1 Begin\n\n",
		         OMPT_MASTER_EV);
	}
	if (inuse[TASKGROUP_START_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		  "0 %d Taskgroup calls\n"
		  "VALUES\n0 Outside\n1 Start\n2 End\n",
		  TASKGROUP_START_EV);
		fprintf (fd, "EVENT_TYPE\n"
		  "0 %d Within Taskgroup region\n"
		  "VALUES\n0 End\n1 Begin\n\n",
		  TASKGROUP_INGROUP_DEEP_EV);
	}
	if (inuse[TASK_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		  "0 %d Task Identifier\n\n",
		  TASKID_EV);
	}
	if (inuse[OMP_STATS_INDEX])
	{
		fprintf (fd,
		  "EVENT_TYPE\n"
		  "0 %d Number of OpenMP instantiated tasks\n"
		  "0 %d Number of OpenMP executed tasks\n\n",
		OMP_STATS_BASE+OMP_NUM_TASKS_INSTANTIATED,
		OMP_STATS_BASE+OMP_NUM_TASKS_EXECUTED);
	}
}

/* New OpenMP support*/
static void NEW_OMPEvent_WriteEnabledOperations (FILE * fd)
{
  int i = 0;

  if (new_inuse[NEW_OMP_CALL_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n");
    fprintf (fd, "0 %d  OpenMP RT Call\n", NEW_OMP_CALL_EV);
    fprintf (fd, "VALUES\n");
    for (i=0; i<MAX_OMP_CALLS; i++)
    {
      fprintf (fd, "%d %s\n", i, OMP_Call_Name[i]);
    }
    fprintf (fd, "\n");
  }
  if (new_inuse[NEW_OMP_NESTED_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n");
    fprintf (fd, "0 %d  OpenMP Nesting Level\n", NEW_OMP_NESTED_EV);
  }
  if (new_inuse[NEW_OMP_PARALLEL_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n");
    fprintf (fd, "0 %d  OpenMP Parallel\n", NEW_OMP_PARALLEL_EV);
    fprintf (fd, "VALUES\n"
                 "0 End\n"
                 "%d PARALLEL REGION Fork/Join\n"
                 "%d PARALLEL REGION\n"
                 "%d LOOP Fork/Join\n"
                 "%d LOOP\n"
                 "%d SECTIONS Fork/Join\n"
                 "%d SECTIONS\n"
                 "%d TEAMS Fork/Join\n"
                 "%d TEAMS\n", 
				 NEW_OMP_PARALLEL_REGION_FORK_VAL, NEW_OMP_PARALLEL_REGION_VAL,
				 NEW_OMP_PARALLEL_LOOP_FORK_VAL, NEW_OMP_PARALLEL_LOOP_VAL,
				 NEW_OMP_PARALLEL_SECTIONS_FORK_VAL, NEW_OMP_PARALLEL_SECTIONS_VAL,
				 NEW_OMP_TEAMS_FORK_VAL, NEW_OMP_TEAMS_VAL);
  }
  if (new_inuse[NEW_OMP_WORKSHARING_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n");
    fprintf (fd, "0 %d OpenMP Worksharing\n", NEW_OMP_WSH_EV);
    fprintf (fd, "VALUES\n");
    fprintf (fd, "0 End\n");
    fprintf (fd, "%d NEXT CHUNK Request\n", NEW_OMP_WSH_NEXT_CHUNK_VAL);
    fprintf (fd, "%d DO/FOR loop\n", NEW_OMP_WSH_DO_VAL);
    fprintf (fd, "%d DO/FOR loop (static)\n", NEW_OMP_WSH_DO_STATIC_VAL);
    fprintf (fd, "%d DO/FOR loop (dynamic)\n", NEW_OMP_WSH_DO_DYNAMIC_VAL);
    fprintf (fd, "%d DO/FOR loop (guided)\n", NEW_OMP_WSH_DO_GUIDED_VAL);
    fprintf (fd, "%d DO/FOR loop (runtime)\n", NEW_OMP_WSH_DO_RUNTIME_VAL);
    fprintf (fd, "%d DO/FOR ordered loop (static)\n", NEW_OMP_WSH_DO_ORDERED_STATIC_VAL);
    fprintf (fd, "%d DO/FOR ordered loop (dynamic)\n", NEW_OMP_WSH_DO_ORDERED_DYNAMIC_VAL);
    fprintf (fd, "%d DO/FOR ordered loop (guided)\n", NEW_OMP_WSH_DO_ORDERED_GUIDED_VAL);
    fprintf (fd, "%d DO/FOR ordered loop (runtime)\n", NEW_OMP_WSH_DO_ORDERED_RUNTIME_VAL);
    fprintf (fd, "%d DOACROSS (static)\n", NEW_OMP_WSH_DOACROSS_STATIC_VAL);
    fprintf (fd, "%d DOACROSS (dynamic)\n", NEW_OMP_WSH_DOACROSS_DYNAMIC_VAL);
    fprintf (fd, "%d DOACROSS (guided)\n", NEW_OMP_WSH_DOACROSS_GUIDED_VAL);
    fprintf (fd, "%d DOACROSS (runtime)\n", NEW_OMP_WSH_DOACROSS_RUNTIME_VAL);
    fprintf (fd, "%d SECTION\n", NEW_OMP_WSH_SECTION_VAL);
    fprintf (fd, "%d SINGLE\n", NEW_OMP_WSH_SINGLE_VAL);
    fprintf (fd, "%d MASTER\n\n", NEW_OMP_WSH_MASTER_VAL);
  }
  if (new_inuse[NEW_OMP_SYNC_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n"
                 "0 %d OpenMP Synchronization\n", NEW_OMP_SYNC_EV);
    fprintf (fd, "VALUES\n"
                 "0 End\n"
                 "%d BARRIER\n"
                 "%d JOIN (wait)\n"
                 "%d JOIN (nowait)\n"
                 "%d ATOMIC\n"
                 "%d CRITICAL\n"
                 "%d CRITICAL (named)\n"
                 "%d ORDERED\n"
                 "%d TASKGROUP\n"
                 "%d TASKWAIT\n"
                 "%d POST\n"
                 "%d WAIT\n\n",
                 NEW_OMP_BARRIER_VAL, NEW_OMP_JOIN_WAIT_VAL, NEW_OMP_JOIN_NOWAIT_VAL,
                 NEW_OMP_LOCK_ATOMIC_VAL, NEW_OMP_LOCK_CRITICAL_VAL, NEW_OMP_LOCK_CRITICAL_NAMED_VAL,
                 NEW_OMP_ORDERED_VAL, NEW_OMP_TASKGROUP_VAL, NEW_OMP_TASKWAIT_VAL, NEW_OMP_POST_VAL, NEW_OMP_WAIT_VAL);
  }
  if (new_inuse[NEW_OMP_LOCK_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n");
    fprintf (fd, "0 %d OpenMP Lock Status\n", NEW_OMP_LOCK_EV);
    fprintf (fd, "VALUES\n"
                 "%d Unlocked\n"
                 "%d LOCK: Obtaining\n"
                 "%d LOCK: Taken\n"
                 "%d LOCK: Releasing\n\n",
                 NEW_OMP_LOCK_RELEASED_VAL, NEW_OMP_LOCK_REQUEST_VAL, NEW_OMP_LOCK_TAKEN_VAL, NEW_OMP_LOCK_RELEASE_REQUEST_VAL);
  }
  if (new_inuse[NEW_OMP_LOCK_NAME_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n"
                 "0 %d OpenMP Lock Address\n", NEW_OMP_LOCK_NAME_EV);
  }
  if (new_inuse[NEW_OMP_ORDERED_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n"
                 "0 %d OpenMP Ordered\n", NEW_OMP_ORDERED_EV);
    fprintf (fd, "VALUES\n"
                 "%d Outside\n"
                 "%d ORDERED: Waiting data\n"
                 "%d ORDERED: Inside\n"
                 "%d ORDERED: Signaling data ready\n\n",
                 NEW_OMP_ORDERED_POST_READY_VAL, NEW_OMP_ORDERED_WAIT_START_VAL, NEW_OMP_ORDERED_WAIT_OVER_VAL, NEW_OMP_ORDERED_POST_START_VAL);
  }
  if (new_inuse[NEW_OMP_TASKGROUP_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n"
                 "0 %d OpenMP Taskgroup\n", NEW_OMP_TASKGROUP_EV);
    fprintf (fd, "VALUES\n"
                 "%d Outside\n"
                 "%d TASKGROUP: Opening\n"
                 "%d TASKGROUP: Inside\n"
                 "%d TASKGROUP: Waiting child\n\n",
                 NEW_OMP_TASKGROUP_END_VAL, NEW_OMP_TASKGROUP_OPENING_VAL, NEW_OMP_TASKGROUP_ENTERING_VAL, NEW_OMP_TASKGROUP_WAITING_VAL);
  }
#if defined(HAVE_BFD)
  if (new_inuse[NEW_OMP_OUTLINED_INDEX])
  {
    Address2Info_Write_OMP_Labels (fd,
      NEW_OMP_OUTLINED_NAME_EV, "OpenMP Outlined function",
      NEW_OMP_OUTLINED_LINE_EV, "OpenMP Outlined function at line/file",
      get_option_merge_UniqueCallerID());
  }
#endif
  if (new_inuse[NEW_OMP_TASKING_INDEX])
  {
    fprintf (fd, "EVENT_TYPE\n"
                 "0 %d OpenMP Tasking\n", NEW_OMP_TASKING_EV);
    fprintf (fd, "VALUES\n"
                 "0 End\n"
                 "%d TASK instantiation\n"
                 "%d TASK execution\n"
                 "%d TASKLOOP instantiation\n"
                 "%d TASKLOOP execution\n\n",
                 NEW_OMP_TASK_INST_VAL, NEW_OMP_TASK_EXEC_VAL, NEW_OMP_TASKLOOP_INST_VAL, NEW_OMP_TASKLOOP_EXEC_VAL);
  }

  if (new_inuse[NEW_OMP_TASK_INST_INDEX])
  {
#if defined(HAVE_BFD)
    Address2Info_Write_OMP_Labels (fd,
      NEW_OMP_TASK_INST_NAME_EV, "OpenMP instantiated task",
      NEW_OMP_TASK_INST_LINE_EV, "OpenMP instantiated task at line/file",
      get_option_merge_UniqueCallerID());
#endif
    fprintf (fd, "EVENT_TYPE\n"
                 "0 %d OpenMP instantiated task ID\n\n",
                 NEW_OMP_TASK_INST_ID_EV);
  }
  if (new_inuse[NEW_OMP_TASK_EXEC_INDEX])
  {
#if defined(HAVE_BFD)
    Address2Info_Write_OMP_Labels (fd,
      NEW_OMP_TASK_EXEC_NAME_EV, "OpenMP executed task",
      NEW_OMP_TASK_EXEC_LINE_EV, "OpenMP executed task at line/file",
      get_option_merge_UniqueCallerID());
#endif
    fprintf (fd, "EVENT_TYPE\n"
                 "0 %d OpenMP executed task ID\n\n",
                 NEW_OMP_TASK_EXEC_ID_EV);
  }
}

void Enable_OMP_Operation (int type)
{
	OLD_Enable_OMP_Operation(type);
	NEW_Enable_OMP_Operation(type);
}

void OMPEvent_WriteEnabledOperations (FILE * fd)
{
	OLD_OMPEvent_WriteEnabledOperations(fd);
	NEW_OMPEvent_WriteEnabledOperations(fd);
}
