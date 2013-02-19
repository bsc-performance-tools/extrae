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
#define TASKWAIT_INDEX         10  /* Taskwait event */

#define MAX_OMP_INDEX		11

static int inuse[MAX_OMP_INDEX] = { FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, 
	FALSE, FALSE, FALSE, FALSE, FALSE };

void Enable_OMP_Operation (int tipus)
{
	if (tipus == PAR_EV)
		inuse[PAR_OMP_INDEX] = TRUE;
	else if (tipus == WSH_EV)
		inuse[WSH_OMP_INDEX] = TRUE;
	else if (tipus == OMPFUNC_EV)
		inuse[FNC_OMP_INDEX] = TRUE;
	else if (tipus == UNNAMEDCRIT_EV)
		inuse[ULCK_OMP_INDEX] = TRUE;
	else if (tipus == NAMEDCRIT_EV)
		inuse[LCK_OMP_INDEX] = TRUE;
	else if (tipus == WORK_EV)
		inuse[WRK_OMP_INDEX] = TRUE;
	else if (tipus == JOIN_EV)
		inuse[JOIN_OMP_INDEX] = TRUE;
	else if (tipus == BARRIEROMP_EV)
		inuse[BARRIER_OMP_INDEX] = TRUE;
	else if (tipus == OMPGETNUMTHREADS_EV || tipus == OMPSETNUMTHREADS_EV)
		inuse[GETSETNUMTHREADS_INDEX] = TRUE;
	else if (tipus == TASK_EV)
		inuse[TASK_INDEX] = TRUE;
	else if (tipus == TASKWAIT_EV)
		inuse[TASKWAIT_INDEX] = TRUE;
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
		fprintf (fd, "%d %d   OpenMP Worksharing join\n", 0, JOIN_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "%d Join (w wait)\n"
		             "%d Join (w/o wait)\n\n",
		             JOIN_WAIT_VAL, JOIN_NOWAIT_VAL);
	}
	if (inuse[WRK_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d  %d  OpenMP Worksharing work dispatcher\n", 0, WORK_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "1 Begin\n\n");
	}
	if (inuse[PAR_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d   %d	 Parallel (OMP)\n", 0, PAR_EV);
		fprintf (fd, "VALUES\n"
		             "0 close\n"
		             "1 DO (open)\n"
		             "2 SECTIONS (open)\n"
		             "3 REGION (open)\n\n");
	}
	if (inuse[WSH_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d   %d	 Worksharing (OMP)\n", 0, WSH_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "4 DO \n"
		             "5 SECTIONS\n"
		             "6 SINGLE\n\n");
	}
#if defined(HAVE_BFD)
	if (inuse[FNC_OMP_INDEX])
	{
		Address2Info_Write_OMP_Labels (fd, OMPFUNC_EV, "Parallel function",
			OMPFUNC_LINE_EV, "Parallel function line and file",
			get_option_merge_UniqueCallerID());
		Address2Info_Write_OMP_Labels (fd, TASKFUNC_INST_EV, "OMP Task function",
			TASKFUNC_INST_LINE_EV, "OMP Task function line and file",
			get_option_merge_UniqueCallerID());
	}
#endif
	if (inuse[LCK_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d   %d	 OpenMP named-Lock\n", 0, NAMEDCRIT_EV);
		fprintf (fd, "VALUES\n"
		             "%d Unlocked status\n"
		             "%d Lock\n"
		             "%d Unlock\n"
		             "%d Locked status\n\n",
		             UNLOCKED_VAL, LOCK_VAL, UNLOCK_VAL, LOCKED_VAL);
	}
	if (inuse[ULCK_OMP_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d   %d	 OpenMP unnamed-Lock\n", 0, UNNAMEDCRIT_EV);
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
		fprintf (fd, "%d   %d OpenMP barrier\n", 0, BARRIEROMP_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "1 Begin\n");
	}
	if (inuse[GETSETNUMTHREADS_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d   %d OpenMP set num threads\n", 0, OMPSETNUMTHREADS_EV);
		fprintf (fd, "%d   %d OpenMP get num threads\n", 0, OMPGETNUMTHREADS_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "1 Begin\n");
	}
	if (inuse[TASK_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d   %d OMP task creation\n", 0, TASK_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "1 Begin\n");
	}
	if (inuse[TASKWAIT_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d   %d OMP task wait\n", 0, TASKWAIT_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n"
		             "1 Begin\n");
	}
}
