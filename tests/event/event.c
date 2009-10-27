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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/tests/event/event.c,v $
 | 
 | @last_commit: $Date: 2008/12/01 10:39:14 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
static char rcsid[] = "$Id: event.c,v 1.4 2008/12/01 10:39:14 gllort Exp $";

#include <stdio.h>

#define MAX_HWC 8
#define HETEROGENEOUS_SUPPORT

typedef struct omp_param_t
{
	unsigned long long param;
} omp_param_t;

typedef struct misc_param_t
{
	unsigned long long param;
} misc_param_t;


typedef struct mpi_param_t
{
	int target;                   /* receiver in send - sender in receive */
	int size;
	int tag;
	int comm;
	int aux;
#if defined(HETEROGENEOUS_SUPPORT)
	int padding[1];
#endif
} mpi_param_t;


typedef union
{
	struct omp_param_t omp_param;
	struct mpi_param_t mpi_param;
	struct misc_param_t misc_param;
} u_param;

/* HSG

  This struct contains the elements of every event that must be recorded.
  The fields must be placed in a such way that the sizeof(event_t) must
  be minimal. Each architecture has it's own preference on the alignament,
  so we must care about the packing of the structure. This is very important
  in the heterogeneous environments.
*/

typedef struct
{
	u_param param;                 /* Parameters of this event              */
	unsigned long long value;      /* Value of this event                   */
	long long time;                /* Timestamp of this event               */
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	long long HWCValues[MAX_HWC];  /* Hardware counters read for this event */
#endif
	int event;                     /* Type of this event                    */
#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)
	int HWCReadSet;                /* Has this event HWC read?              */
#endif
} event_t;

int main (int argc, char *argv[])
{
	event_t e;

	printf ("\nsizeof(event_t) = %d\n\n", sizeof(event_t));
	printf ("@e.param        = %d\n", ((long) &e.param) - ((long) &e));
	printf ("@e.value        = %d\n", ((long) &e.value) - ((long) &e));
	printf ("@e.time         = %d\n", ((long) &e.time) - ((long) &e));
	printf ("@e.HWCValues    = %d\n", ((long) &e.HWCValues) - ((long) &e));
	printf ("@e.event        = %d\n", ((long) &e.event) - ((long) &e));
	printf ("@e.HWCReadSet   = %d\n", ((long) &e.HWCReadSet) - ((long) &e));
	printf ("\n");

	return 0;
}

