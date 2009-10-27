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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/trace_to_prv.h,v $
 | 
 | @last_commit: $Date: 2008/10/24 08:11:22 $
 | @version:     $Revision: 1.7 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef TRACE_TO_PRV_H
#define TRACE_TO_PRV_H

#include "mpi2out.h"
#ifdef HAVE_ZLIB
# include "zlib.h"
#endif

#include "cpunode.h"
#include "fdz.h"

extern struct ptask_t *obj_table;
extern unsigned int num_ptasks;

int Paraver_ProcessTraceFiles (char *prvName, unsigned long nfiles,
	struct input_t *files, unsigned int num_appl, char *sym_file,
	struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int idtask,
	int MBytesPerAllSegments, int forceformat);

extern int **EnabledTasks;
extern unsigned long long **EnabledTasks_time;

void AnotaBGLPersonality (unsigned int event, unsigned long long valor, int task);

#endif /* __TRACE_TO_PRV_H__ */
