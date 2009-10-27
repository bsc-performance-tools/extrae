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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/misc_prv_events.c,v $
 | 
 | @last_commit: $Date: 2009/04/21 10:40:40 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: misc_prv_events.c,v 1.4 2009/04/21 10:40:40 gllort Exp $";

#include <config.h>

#if defined(HAVE_STDLIB_H)
# include <stdlib.h>
#endif

#include "misc_prv_events.h"
#include "labels.h"

static int TOPOLOGY_in_use = FALSE;	/* PARALLEL constructs */

void Enable_Topology ()
{
	TOPOLOGY_in_use = TRUE;
}

void MISCEvent_WriteEnabledOperations (FILE * fd)
{
#if defined(IS_MN_MACHINE)
	if (TOPOLOGY_in_use)
	{
    fprintf (fd, "EVENT_TYPE\n");
    fprintf (fd, "%d   %d   %s\n", 0, LINEAR_HOST_EVENT, LINEAR_HOST_LABEL);
    fprintf (fd, "%d   %d   %s\n", 0, LINECARD_EVENT, LINECARD_LABEL);
    fprintf (fd, "%d   %d   %s\n", 0, HOST_EVENT, HOST_LABEL);
		LET_SPACES(fd);
	}
#endif
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"
#include "mpi-tags.h"

void Share_MISC_Operations (void)
{
	int res, i, max;
	int tmp2[3], tmp[3] = { TOPOLOGY_in_use, Rusage_Events_Found, MPIStats_Events_Found };
	int tmp_in[RUSAGE_EVENTS_COUNT], tmp_out[RUSAGE_EVENTS_COUNT];
	int tmp2_in[MPI_STATS_EVENTS_COUNT], tmp2_out[MPI_STATS_EVENTS_COUNT];

	res = MPI_Reduce (tmp, tmp2, 3, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #1");

	TOPOLOGY_in_use = tmp2[0];
	Rusage_Events_Found = tmp2[1];
	MPIStats_Events_Found = tmp2[2];

	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		tmp_in[i] = GetRusage_Labels_Used[i];

	res = MPI_Reduce (tmp_in, tmp_out, RUSAGE_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #2");
	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		GetRusage_Labels_Used[i] = tmp_out[i];

	for (i = 0; i < MPI_STATS_EVENTS_COUNT; i++)
		tmp2_in[i] = MPIStats_Labels_Used[i];

	res = MPI_Reduce (tmp2_in, tmp2_out, MPI_STATS_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #3");
	for (i = 0; i < MPI_STATS_EVENTS_COUNT; i++)
		MPIStats_Labels_Used[i] = tmp2_out[i];

	res = MPI_Reduce (&MaxClusterId, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #4");
	MaxClusterId = max;
}

#endif
