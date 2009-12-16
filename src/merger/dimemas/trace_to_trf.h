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

#ifndef TRACE_TO_TRF_H
#define TRACE_TO_TRF_H

#include "mpi2out.h"
#include "cpunode.h"
#include "fdz.h"

extern struct ptask_t *obj_table;
extern unsigned int num_ptasks;

int Dimemas_ProcessTraceFiles (char *prvName, unsigned long nfiles,
	struct input_t *files, unsigned int num_appl, char *callback_file,
	struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int idtask,
	int MBytesPerAllSegments, int forceformat);

unsigned long long Dimemas_hr_to_relative (UINT64 iotimer);

#if defined(DEAD_CODE)
extern int **EnabledTasks;
extern unsigned long long **EnabledTasks_time;
#endif

void AnotaBGPersonality (unsigned int event, unsigned long long valor, int task);

#endif /* __TRACE_TO_PRV_H__ */
