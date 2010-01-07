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

#include <config.h>

#if defined(HAVE_STDLIB_H)
# include <stdlib.h>
#endif

#include "misc_prv_events.h"
#include "misc_prv_semantics.h"
#include "labels.h"

void MISCEvent_WriteEnabledOperations (FILE * fd, long long options)
{
	if (options & TRACEOPTION_MN_ARCH)
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d   %d   %s\n", 0, MN_LINEAR_HOST_EVENT, MN_LINEAR_HOST_LABEL);
		fprintf (fd, "%d   %d   %s\n", 0, MN_LINECARD_EVENT, MN_LINECARD_LABEL);
		fprintf (fd, "%d   %d   %s\n", 0, MN_HOST_EVENT, MN_HOST_LABEL);
		LET_SPACES(fd);
	}
	else if (options & TRACEOPTION_BG_ARCH)
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_X, BG_TORUS_X);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_Y, BG_TORUS_Y);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_Z, BG_TORUS_Z);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_PROCESSOR_ID, BG_PROCESSOR_ID);
		LET_SPACES (fd);
	}
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"
#include "mpi-tags.h"

void Share_MISC_Operations (void)
{
	int res, i, max;
	int tmp2[3], tmp[3] = { Rusage_Events_Found, MPI_Stats_Events_Found, PACX_Stats_Events_Found };
	int tmp_in[RUSAGE_EVENTS_COUNT], tmp_out[RUSAGE_EVENTS_COUNT];
	int tmp2_in[MPI_STATS_EVENTS_COUNT], tmp2_out[MPI_STATS_EVENTS_COUNT];
	int tmp3_in[PACX_STATS_EVENTS_COUNT], tmp3_out[PACX_STATS_EVENTS_COUNT];

	res = MPI_Reduce (tmp, tmp2, 3, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #1");
	Rusage_Events_Found = tmp2[0];
	MPI_Stats_Events_Found = tmp2[1];
	PACX_Stats_Events_Found = tmp2[2];

	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		tmp_in[i] = GetRusage_Labels_Used[i];
	res = MPI_Reduce (tmp_in, tmp_out, RUSAGE_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #2");
	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		GetRusage_Labels_Used[i] = tmp_out[i];

	for (i = 0; i < MPI_STATS_EVENTS_COUNT; i++)
		tmp2_in[i] = MPI_Stats_Labels_Used[i];
	res = MPI_Reduce (tmp2_in, tmp2_out, MPI_STATS_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #3");
	for (i = 0; i < MPI_STATS_EVENTS_COUNT; i++)
		MPI_Stats_Labels_Used[i] = tmp2_out[i];

	for (i = 0; i < PACX_STATS_EVENTS_COUNT; i++)
		tmp3_in[i] = PACX_Stats_Labels_Used[i];
	res = MPI_Reduce (tmp3_in, tmp3_out, PACX_STATS_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #4");
	for (i = 0; i < PACX_STATS_EVENTS_COUNT; i++)
		PACX_Stats_Labels_Used[i] = tmp3_out[i];

	res = MPI_Reduce (&MaxClusterId, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #5");
	MaxClusterId = max;
}

#endif
