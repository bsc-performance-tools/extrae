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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/omp_prv_events.c,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:06:55 $
 | @version:     $Revision: 1.6 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: omp_prv_events.c,v 1.6 2009/05/28 13:06:55 harald Exp $";

#ifdef HAVE_BFD
# include "addr2info.h"
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "events.h"
#include "omp_prv_events.h"
#include "mpi2out.h"

#define PAR_OMP_INDEX           0  /* PARALLEL constructs */
#define WSH_OMP_INDEX           1  /* WORKSHARING constructs */
#define FNC_OMP_INDEX           2  /* Pointers to routines <@> */
#define ULCK_OMP_INDEX          3  /* Unnamed locks in use! */
#define LCK_OMP_INDEX           4  /* Named locks in use! */
#define WRK_OMP_INDEX           5  /* Work delivery */
#define JOIN_OMP_INDEX          6  /* Joins */

#define MAX_OMP_INDEX		7

static int inuse[MAX_OMP_INDEX] = { FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE };

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
		Address2Info_Write_OMP_Labels (fd, OMPFUNC_EV, OMPFUNC_LINE_EV, option_UniqueCallerID);
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
}
