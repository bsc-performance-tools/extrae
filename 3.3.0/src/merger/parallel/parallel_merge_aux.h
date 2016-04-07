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

#ifndef PARALLEL_MERGE_AUX_H
#define PARALLEL_MERGE_AUX_H

#include <config.h>

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "common.h"
#include "file_set.h"

struct ForeignRecv_t
{
	UINT64 physic, logic;
	int sender, sender_app, recver, recver_app, tag, match_zone;
	unsigned thread, vthread;
};
struct ForeignRecvs_t
{
	int count, size;
	struct ForeignRecv_t *data;
};

void InitForeignRecvs (int numtasks);
void AddForeignRecv (UINT64 physic, UINT64 logic, int tag, int ptask_r, int task_r,
	unsigned thread_r, unsigned vthread_r, int ptask_s, int task_s, FileSet_t *fset, int mz);

void DistributePendingComms (int numtasks, int taskid);
void NewDistributePendingComms (int numtasks, int taskid, int match);
struct ForeignRecv_t* SearchForeignRecv (int group, int sender_app, int sender, int recver_app, int recver, int tag, int mz);

void AddPendingCommunication (int descriptor, off_t offset, int tag, int task_r,
	int task_s, int mz);
void InitPendingCommunication (void);

void ParallelMerge_InitCommunicators(void);
void ParallelMerge_AddIntraCommunicator (int ptask, int task, int type, int id,
	int ntasks, int *tasks);
void ParallelMerge_AddInterCommunicator (int ptask, int task, int id, int comm1,
	int leader1, int comm2, int leader2);
void ParallelMerge_BuildCommunicators (int num_tasks, int taskid);

void ShareNodeNames (int numtasks, char ***nodenames);

void ShareTraceInformation (int numtasks, int taskid);

void Gather_Dimemas_Traces (int numtasks, int taskid, FILE *fd, unsigned int maxmem);
void Gather_Dimemas_Offsets (int numtasks, int taskid, int count,
	unsigned long long *in_offsets, unsigned long long **out_offsets,
	unsigned long long local_trace_size, FileSet_t *fset);

unsigned * Gather_Paraver_VirtualThreads (unsigned taskid, unsigned ptask,
	FileSet_t *fset);

#endif /* PARALLEL_MERGE_AUX_H */
