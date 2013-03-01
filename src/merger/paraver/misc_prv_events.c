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

#define APPL_INDEX              0
#define FLUSH_INDEX             1
#define TRACING_INDEX           2
#define INOUT_INDEX             3
#define FORK_INDEX              4
#define GETCPU_INDEX            5

#define MAX_MISC_INDEX	        6

static int inuse[MAX_MISC_INDEX] = { FALSE, FALSE, FALSE, FALSE, FALSE, FALSE };

void Enable_MISC_Operation (int type)
{
	if (type == APPL_EV)
		inuse[APPL_INDEX] = TRUE;
	else if (type == FLUSH_EV)
		inuse[FLUSH_INDEX] = TRUE;
	else if (type == TRACING_EV)
		inuse[TRACING_INDEX] = TRUE;
	else if (type == READ_EV || type == WRITE_EV || type == IOSIZE_EV)
		inuse[INOUT_INDEX] = TRUE;
	else if (type == FORK_EV || type == WAIT_EV || type == WAITPID_EV || type == EXEC_EV)
		inuse[FORK_INDEX] = TRUE;
	else if (type == GETCPU_EV)
		inuse[GETCPU_INDEX] = TRUE;
}

void MISCEvent_WriteEnabledOperations (FILE * fd, long long options)
{	
	if (options & TRACEOPTION_BG_ARCH)
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_PROCESSOR_ID, BG_PROCESSOR_ID);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_A, BG_TORUS_A);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_B, BG_TORUS_B);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_C, BG_TORUS_C);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_D, BG_TORUS_D);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, BG_PERSONALITY_TORUS_E, BG_TORUS_E);
		LET_SPACES (fd);
	}
	if (inuse[GETCPU_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, GETCPU_EV, GETCPU_LBL);
		LET_SPACES(fd);
	}
	if (inuse[APPL_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, APPL_EV, APPL_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, EVT_BEGIN_LBL);
		LET_SPACES (fd);
	}
	if (inuse[FLUSH_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, FLUSH_EV, FLUSH_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, EVT_BEGIN_LBL);
		LET_SPACES (fd);
	}
	if (inuse[TRACING_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, TRACING_EV, TRACING_LBL);

		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, TRAC_DISABLED_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, TRAC_ENABLED_LBL);
		LET_SPACES (fd);
	}
	if (inuse[INOUT_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, READ_EV, READ_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, WRITE_EV, WRITE_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, EVT_BEGIN_LBL);

		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, IOSIZE_EV, IOSIZE_LBL);
		LET_SPACES (fd);
	}
	if (inuse[FORK_INDEX])
	{
		fprintf (fd, "%s\n", TYPE_LABEL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, FORK_EV, FORK_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, WAIT_EV, WAIT_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, WAITPID_EV, WAITPID_LBL);
		fprintf (fd, "%d    %d    %s\n", MISC_GRADIENT, EXEC_EV, EXEC_LBL);
		fprintf (fd, "%s\n", VALUES_LABEL);
		fprintf (fd, "%d      %s\n", EVT_END, EVT_END_LBL);
		fprintf (fd, "%d      %s\n", EVT_BEGIN, EVT_BEGIN_LBL);
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
	int tmp_misc[MAX_MISC_INDEX];

	res = MPI_Reduce (inuse, tmp_misc, MAX_MISC_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #1");
	for (i = 0; i < MAX_MISC_INDEX; i++)
		inuse[i] = tmp_misc[i];

	res = MPI_Reduce (tmp, tmp2, 4, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #2");
	Rusage_Events_Found = tmp2[0];
	MPI_Stats_Events_Found = tmp2[1];
	PACX_Stats_Events_Found = tmp2[2];
	Memusage_Events_Found = tmp2[3];

	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		tmp_in[i] = GetRusage_Labels_Used[i];
	res = MPI_Reduce (tmp_in, tmp_out, RUSAGE_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #3");
	for (i = 0; i < RUSAGE_EVENTS_COUNT; i++)
		GetRusage_Labels_Used[i] = tmp_out[i];

	for (i = 0; i < MPI_STATS_EVENTS_COUNT; i++)
		tmp2_in[i] = MPI_Stats_Labels_Used[i];
	res = MPI_Reduce (tmp2_in, tmp2_out, MPI_STATS_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #4");
	for (i = 0; i < MPI_STATS_EVENTS_COUNT; i++)
		MPI_Stats_Labels_Used[i] = tmp2_out[i];

	for (i = 0; i < PACX_STATS_EVENTS_COUNT; i++)
		tmp3_in[i] = PACX_Stats_Labels_Used[i];
	res = MPI_Reduce (tmp3_in, tmp3_out, PACX_STATS_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #5");
	for (i = 0; i < PACX_STATS_EVENTS_COUNT; i++)
		PACX_Stats_Labels_Used[i] = tmp3_out[i];

	for (i = 0; i < MEMUSAGE_EVENTS_COUNT; i++)
		tmp4_in[i] = Memusage_Labels_Used[i];
	res = MPI_Reduce (tmp4_in, tmp4_out, MEMUSAGE_EVENTS_COUNT, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #6");
	for (i = 0; i < MEMUSAGE_EVENTS_COUNT; i++)
		Memusage_Labels_Used[i] = tmp4_out[i];

	res = MPI_Reduce (&MaxClusterId, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing MISC operations #7");
	MaxClusterId = max;
}

#endif
