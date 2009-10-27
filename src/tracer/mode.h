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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/mode.h,v $
 | 
 | @last_commit: $Date: 2008/01/03 11:56:55 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __MODE_H__
#define __MODE_H__

#include "clock.h"
#include "trace_mode.h"

extern int *MPI_Deepness;
extern int *Current_Trace_Mode;
extern int *Pending_Trace_Mode_Change;

#define CURRENT_TRACE_MODE(tid) Current_Trace_Mode[tid]
#define PENDING_TRACE_MODE_CHANGE(tid) Pending_Trace_Mode_Change[tid]

#define INCREASE_MPI_DEEPNESS(tid) (MPI_Deepness[tid]++)
#define DECREASE_MPI_DEEPNESS(tid) (MPI_Deepness[tid]--)
#define MPI_IS_NOT_STACKED(tid) (MPI_Deepness[tid] == 0)

void TMODE_setInitial (int mode);
int Trace_Mode_Initialize (int num_threads);
int Trace_Mode_reInitialize (int old_num_threads, int new_num_threads);
void Trace_Mode_Change (int tid, iotimer_t time);

/* Bursts mode specific */

extern unsigned long long BurstsMode_Threshold;
extern int BurstsMode_MPI_Stats;

#define MINIMUM_BURST_DURATION (BurstsMode_Threshold)
#define TRACING_MPI_STATISTICS (BurstsMode_MPI_Stats)

void TMODE_setBurstsThreshold  (unsigned long long threshold);
void TMODE_setBurstsStatistics (int status);

#endif /* __MODE_H__ */
