/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
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

#ifndef _MPI2OUT_H
#define _MPI2OUT_H

#include "config.h"

#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif

extern unsigned nTraces;

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

	unsigned long long SpawnOffset;

	int fd;
	char *name;
	char *node;
	char *threadname;
}
input_t;

#define GetInput_ptask(item)  ((item)->ptask)
#define GetInput_task(item)   ((item)->task)
#define GetInput_name(item)   ((item)->name)
#define GetInput_fd(item)     ((item)->fd)

typedef enum {FileOpen_Default, FileOpen_Absolute, FileOpen_Relative} FileOpen_t;

void merger_pre (int numtasks);
void ProcessArgs (int rank, int argc, char *argv[]);
int merger_post (int numtasks, int idtask);

void Read_MPITS_file (const char *file, int *cptask, FileOpen_t opentype, int taskid);

#endif
