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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/trt_prv_events.c,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:06:55 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: trt_prv_events.c,v 1.3 2009/05/28 13:06:55 harald Exp $";

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
