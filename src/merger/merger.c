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

#include "mpi2out.h"

#if defined(PARALLEL_MERGE)

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include <mpi.h>
#include "mpi-aux.h"

int main (int argc, char *argv[])
{
	int res;
	int ntasks;
	int idtask;

	res = MPI_Init (&argc, &argv);
	MPI_CHECK (res, MPI_Init, "Failed to initialize MPI");

	res = MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
	MPI_CHECK (res, MPI_Comm_size, "Failed to call MPI_Comm_size");

	res = MPI_Comm_rank (MPI_COMM_WORLD, &idtask);
	MPI_CHECK (res, MPI_Comm_size, "Failed to call MPI_Comm_rank");

	merger (ntasks, idtask, argc, argv);

	res = MPI_Finalize ();
	MPI_CHECK (res, MPI_Finalize, "Failed to uninitialize MPI");

	return 0;
}
#else
int main (int argc, char *argv[])
{
	return merger (1, 0, argc, argv);
}
#endif

