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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/common/mpi2out.h,v $
 | 
 | @last_commit: $Date: 2009/05/28 13:06:55 $
 | @version:     $Revision: 1.10 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _MPI2OUT_H
#define _MPI2OUT_H

#include "config.h"

#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif

typedef struct input_t
{
	off_t filesize;
	unsigned int order;
	unsigned int cpu;
	unsigned int nodeid;
	unsigned int ptask;
	unsigned int task;
	unsigned int thread;

	int InputForWorker;           /* Which task is responsible for this file */

#if defined(DEAD_CODE)
#define FD_TYPE 0
#define NAME_TYPE 1
	unsigned int type;            /* Tells if file is given as an fd or as a name */
#endif

	int fd;
	char *name;
	char *node;
}
input_t;

#define GetInput_ptask(item)  ((item)->ptask)
#define GetInput_task(item)   ((item)->task)
#define GetInput_name(item)   ((item)->name)
#define GetInput_fd(item)     ((item)->fd)

#if defined(IS_BG_MACHINE)    /* BlueGene coordinates are kept in traces */
extern int option_XYZT;
#endif

extern int SincronitzaTasks;
extern int SincronitzaTasks_byNode;
extern int dump;
extern int Joint_States;
extern int option_UseDiskForComms;
extern int option_SkipSendRecvComms;
extern int option_UniqueCallerID;

int merger (int numtasks, int idtask, int argc, char *argv[]);

#endif
