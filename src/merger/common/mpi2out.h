/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
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
