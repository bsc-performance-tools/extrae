/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mrnet/libextract/LongGlops.h,v $
 | 
 | @last_commit: $Date: 2009/04/21 10:41:31 $
 | @version:     $Revision: 1.1 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __LONG_GLOPS_H__
#define __LONG_GLOPS_H__

#if defined(__cplusplus)
extern "C" {
#endif

int Extract_LongGlops (int task_id, int thread_id, unsigned long long **io_Glops_Durations, int *io_firstGlopID, int *io_lastGlopID);
void Filter_Long_Glops (int task_id, int thread_id, int commonFirstGlop, int commonLastGlop, unsigned int *Selected_Glops);

#if defined(__cplusplus)
}
#endif

#include <mpi.h>
extern int Is_MPI_World_Comm (MPI_Comm comm);

#endif /* __LONG_GLOPS_H__ */
