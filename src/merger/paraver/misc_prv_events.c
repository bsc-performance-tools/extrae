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
	int tmp2[4], tmp[4] = { Rusage_Events_Found, MPI_Stats_Events_Found, PACX_Stats_Events_Found, Memusage_Events_Found };
	int tmp_in[RUSAGE_EVENTS_COUNT], tmp_out[RUSAGE_EVENTS_COUNT];
	int tmp2_in[MPI_STATS_EVENTS_COUNT], tmp2_out[MPI_STATS_EVENTS_COUNT];
	int tmp3_in[PACX_STATS_EVENTS_COUNT], tmp3_out[PACX_STATS_EVENTS_COUNT];
	int tmp4_in[MEMUSAGE_EVENTS_COUNT], tmp4_out[MEMUSAGE_EVENTS_COUNT];

	res = MPI_Reduce (tmp, tmp2, 4, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #1");
	Rusage_Events_Found = tmp2[0];
	MPI_Stats_Events_Found = tmp2[1];
	PACX_Stats_Events_Found = tmp2[2];
	Memusage_Events_Found = tmp2[3];

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

	for (i = 0; i < MEMUSAGE_EVENTS_COUNT; i++)
		tmp4_in[i] = Memusage_Labels_Used[i];
	res = MPI_Reduce (tmp4_in, tmp4_out, MEMUSAGE_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #5");
	for (i = 0; i < MEMUSAGE_EVENTS_COUNT; i++)
		Memusage_Labels_Used[i] = tmp4_out[i];

	res = MPI_Reduce (&MaxClusterId, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #6");
	MaxClusterId = max;
}

#endif
