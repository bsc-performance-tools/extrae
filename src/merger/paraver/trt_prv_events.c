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
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_BFD
# include "addr2info.h"
#endif

#include "events.h"
#include "trt_prv_events.h"
#include "mpi2out.h"

#define TRT_SPAWN_INDEX       0  /* threadSpawn index */
#define TRT_READ_INDEX        1  /* threadRead index */
#define TRT_USR_FUNC_INDEX    2  /* pthread_create @ target address index */

#define MAX_TRT_INDEX         3

static int inuse[MAX_TRT_INDEX] = { FALSE, FALSE, FALSE };

void Enable_TRT_Operation (int tipus)
{
	if (tipus == TRT_SPAWN_EV)
		inuse[TRT_SPAWN_INDEX] = TRUE;
	else if (tipus == TRT_READ_EV)
		inuse[TRT_READ_INDEX] = TRUE;
	else if (tipus == TRT_USRFUNC_EV)
		inuse[TRT_USR_FUNC_INDEX] = TRUE;
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_TRT_Operations (void)
{
	int res, i, tmp[MAX_TRT_INDEX];

	res = MPI_Reduce (inuse, tmp, MAX_TRT_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing TRT enabled operations");

	for (i = 0; i < MAX_TRT_INDEX; i++)
		inuse[i] = tmp[i];
}

#endif

void TRTEvent_WriteEnabledOperations (FILE * fd)
{
	if (inuse[TRT_SPAWN_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    threadSpawn\n", 0, TRT_SPAWN_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[TRT_READ_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    threadRead\n", 0, TRT_READ_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
#if defined(HAVE_BFD)
	/* Hey, pthread & OpenMP share the same labels? */
	if (inuse[TRT_USR_FUNC_INDEX])
		Address2Info_Write_OMP_Labels (fd, PTHREADFUNC_EV, PTHREADFUNC_LINE_EV, option_UniqueCallerID);
#endif
}
