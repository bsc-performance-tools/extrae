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

#ifndef __MISC_PRV_SEMANTICS_H__
#define __MISC_PRV_SEMANTICS_H__

#include "record.h"
#include "semantics.h"
#include "file_set.h"

int MPI_Caller_Multiple_Levels_Traced;
int *MPI_Caller_Labels_Used;

int Sample_Caller_Multiple_Levels_Traced;
int *Sample_Caller_Labels_Used;

int Rusage_Events_Found;
int GetRusage_Labels_Used[RUSAGE_EVENTS_COUNT];

int MPI_Stats_Events_Found;
int MPI_Stats_Labels_Used[MPI_STATS_EVENTS_COUNT];

int PACX_Stats_Events_Found;
int PACX_Stats_Labels_Used[PACX_STATS_EVENTS_COUNT];

extern SingleEv_Handler_t PRV_MISC_Event_Handlers[];
extern RangeEv_Handler_t PRV_MISC_Range_Handlers[];

int HWC_Change_Ev (
   int newSet,
   unsigned long long current_time,
   unsigned int cpu,
   unsigned int ptask,
   unsigned int task,
   unsigned int thread);

#endif /* __MISC_PRV_SEMANTICS_H__ */
