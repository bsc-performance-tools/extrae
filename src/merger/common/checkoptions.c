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
