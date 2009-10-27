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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/pthread_prv_events.c,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:06:55 $
 | @version:     $Revision: 1.5 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: pthread_prv_events.c,v 1.5 2009/05/28 13:06:55 harald Exp $";

#ifdef HAVE_BFD
# include "addr2info.h"
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "events.h"
#include "omp_prv_events.h"
#include "mpi2out.h"

#define PTHD_CREATE_INDEX       0  /* pthread_create index */
#define PTHD_JOIN_INDEX         1  /* pthread_join index */
#define PTHD_DETACH_INDEX       2  /* pthread_detach index */
#define PTHD_USRF_INDEX         3  /* pthread_create @ target address index */

#define MAX_PTHD_INDEX		4

static int inuse[MAX_PTHD_INDEX] = { FALSE, FALSE, FALSE, FALSE };

void Enable_pthread_Operation (int tipus)
{
	if (tipus == PTHREADCREATE_EV)
		inuse[PTHD_CREATE_INDEX] = TRUE;
	else if (tipus == PTHREADJOIN_EV)
		inuse[PTHD_JOIN_INDEX] = TRUE;
	else if (tipus == PTHREADDETACH_EV)
		inuse[PTHD_DETACH_INDEX] = TRUE;
	else if (tipus == PTHREADFUNC_EV)
		inuse[PTHD_USRF_INDEX] = TRUE;
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_pthread_Operations (void)
{
	int res, i, tmp[MAX_PTHD_INDEX];

	res = MPI_Reduce (inuse, tmp, MAX_PTHD_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing pthread enabled operations");

	for (i = 0; i < MAX_PTHD_INDEX; i++)
		inuse[i] = tmp[i];
}

#endif

void pthreadEvent_WriteEnabledOperations (FILE * fd)
{
	if (inuse[PTHD_CREATE_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_create\n", 0, PTHREADCREATE_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[PTHD_JOIN_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_join\n", 0, PTHREADJOIN_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
	if (inuse[PTHD_DETACH_INDEX])
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    pthread_detach\n", 0, PTHREADDETACH_EV);
		fprintf (fd, "VALUES\n0 End\n1 Begin\n\n");
	}
#if defined(HAVE_BFD)
	/* Hey, pthread & OpenMP share the same labels? */
	if (inuse[PTHD_USRF_INDEX])
		Address2Info_Write_OMP_Labels (fd, PTHREADFUNC_EV, PTHREADFUNC_LINE_EV, option_UniqueCallerID);
#endif
}
