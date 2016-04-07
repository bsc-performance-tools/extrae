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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if defined(PARALLEL_MERGE)
# include <mpi.h>
# include "mpi-aux.h"
#endif

#include "semantics.h"

void CheckHWCcontrol (int taskid, long long options)
{
	int canproceed = FALSE;
# if defined(PARALLEL_MERGE)
	int res;
#endif
#if !defined(HETEROGENEOUS_SUPPORT)
	unsigned int use_hwc = FALSE;
# if USE_HARDWARE_COUNTERS
	unsigned int use_hardware_counters = TRUE;
# else
	unsigned int use_hardware_counters = FALSE;
# endif
#endif

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Hardware Counters control... ");
		fflush (stdout);

#if defined(HETEROGENEOUS_SUPPORT)
		fprintf (stdout, "implicitly defined by heterogeneous support!\n");
		canproceed = TRUE;
#else

		use_hwc = (options & TRACEOPTION_HWC)?TRUE:FALSE;
		canproceed = (use_hwc == use_hardware_counters);

		if (!canproceed)
		{
			fprintf (stdout, " FAILED! Dying...\n");
			fflush (stdout);
		}
		else
		{
			fprintf (stdout, " passed!\n");
			fflush (stdout);
		}
#endif /* HETEROGENEOUS_SUPPORT */
	} /* taskid == 0 */

#if defined(PARALLEL_MERGE)
	res = MPI_Bcast (&canproceed, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to share CheckHWCcontrol result!");
#endif

	if (!canproceed)
	{
#if defined(PARALLEL_MERGE)
		MPI_Finalize();
#endif
		exit (-1);
	}
}

void CheckClockType (int taskid, long long options, int traceformat, int force)
{
# if defined(PARALLEL_MERGE)
	int res;
#endif
	int canproceed = FALSE;
	int trace_dimemas = (options & TRACEOPTION_DIMEMAS)?TRUE:FALSE;

	if (taskid == 0)
	{
		fprintf (stdout, "mpi2prv: Selected output trace format is %s\n", traceformat==PRV_SEMANTICS?"Paraver":"Dimemas");
		fprintf (stdout, "mpi2prv: Stored trace format is %s\n", trace_dimemas?"Dimemas":"Paraver");
		fflush (stdout);

		if ((!trace_dimemas &&  (traceformat == TRF_SEMANTICS)) ||
		    (trace_dimemas && (traceformat == PRV_SEMANTICS)))
		{
			if (!force)
			{
				fprintf (stderr, "mpi2prv: ERROR! Trace Input & Output format mismatch!\n");
				fprintf (stderr, "mpi2prv:        Input is %s whereas output is %s\n", trace_dimemas?"Dimemas":"Paraver", (traceformat==PRV_SEMANTICS)?"Paraver":"Dimemas");
				fflush (stderr);
				canproceed = FALSE;
			}
			else
			{
				fprintf (stderr, "mpi2prv: WARNING! Trace Input & Output format mismatch!\n");
				fprintf (stderr, "mpi2prv:          Input is %s whereas output is %s\n", trace_dimemas?"Dimemas":"Paraver", (traceformat==PRV_SEMANTICS)?"Paraver":"Dimemas");
				fflush (stderr);
				canproceed = TRUE;
			}
			
		}
		else
			canproceed = TRUE;
	} /* taskid == 0 */

#if defined(PARALLEL_MERGE)
	res = MPI_Bcast (&canproceed, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Bcast, "Failed to share Clock/Trace Type result!");
#endif

	if (!canproceed)
	{
#if defined(PARALLEL_MERGE)
		MPI_Finalize();
#endif
		exit (-1);
	}
}
