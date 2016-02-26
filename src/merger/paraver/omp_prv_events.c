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

#define MAX_OMP_INDEX		    20

static int inuse[MAX_OMP_INDEX] = { FALSE };

void Enable_OMP_Operation (int type)
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
	int res, i, tmp[MAX_OMP_INDEX];

	res = MPI_Reduce (inuse, tmp, MAX_OMP_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing OpenMP enabled operations");

	for (i = 0; i < MAX_OMP_INDEX; i++)
		inuse[i] = tmp[i];
}

#endif

void OMPEvent_WriteEnabledOperations (FILE * fd)
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
		fprintf (fd, "0 %d OMP task wait\n", TASKWAIT_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
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
