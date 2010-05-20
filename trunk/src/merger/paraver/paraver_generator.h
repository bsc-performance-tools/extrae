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

#ifndef __PARAVER_GENERATOR_H__
#define __PARAVER_GENERATOR_H__

#include "mpi2out.h"
#include "trace_to_prv.h"

void trace_paraver_state (unsigned int cpu, unsigned int ptask, 
	unsigned int task, unsigned int thread, 
	unsigned long long current_time);

void trace_paraver_event (unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, unsigned long long time, 
	unsigned int type, UINT64 value);

void trace_paraver_unmatched_communication (unsigned int cpu_s, unsigned int ptask_s,
	unsigned int task_s, unsigned int thread_s, unsigned long long log_s,
	unsigned long long phy_s, unsigned int cpu_r, unsigned int ptask_r,
	unsigned int task_r, unsigned int thread_r, unsigned int size, unsigned int tag);

void trace_paraver_communication (unsigned int cpu_s, unsigned int ptask_s,
	unsigned int task_s, unsigned int thread_s, unsigned long long log_s,
	unsigned long long phy_s, unsigned int cpu_r, unsigned int ptask_r,
	unsigned int task_r, unsigned int thread_r, unsigned long long log_r,
	unsigned long long phy_r, unsigned int size, unsigned int tag,
	int givenOffset, off_t position);

int trace_paraver_pending_communication (unsigned int cpu_s, 
	unsigned int ptask_s, unsigned int task_s, unsigned int thread_s,
	unsigned long long log_s, unsigned long long phy_s, unsigned int cpu_r, 
	unsigned int ptask_r, unsigned int task_r, unsigned int thread_r,
	unsigned long long log_r, unsigned long long phy_r, unsigned int size,
	unsigned int tag);

void trace_enter_global_op (unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, unsigned long long time, 
	unsigned int com_id, unsigned int send_size, unsigned int recv_size,
	unsigned int is_root, unsigned isMPI);

#if defined(DEAD_CODE)
void trace_leave_global_op (unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, unsigned long long time,
	int trace_size);
#endif

int paraver_state (struct fdz_fitxer fdz, unsigned int cpu,
	unsigned int ptask, unsigned int task, unsigned int thread,
	unsigned long long ini_time, unsigned long long end_time,
	unsigned int state);

int paraver_event (struct fdz_fitxer fdz, unsigned int cpu,
	unsigned int ptask, unsigned int task, unsigned int thread,
	unsigned long long time, unsigned int type, UINT64 value);

int paraver_multi_event (struct fdz_fitxer fdz, unsigned int cpu,
	unsigned int ptask, unsigned int task, unsigned int thread,
	unsigned long long time, unsigned int count, unsigned int *type,
	UINT64 *value);

int paraver_communication (struct fdz_fitxer fdz, unsigned int cpu_s,
	unsigned int ptask_s, unsigned int task_s, unsigned int thread_s,
	unsigned long long log_s, unsigned long long phy_s, unsigned int cpu_r,
	unsigned int ptask_r, unsigned int task_r, unsigned int thread_r,
	unsigned long long log_r, unsigned long long phy_r, unsigned int size,
	unsigned int tag);

int paraver_global_op (struct fdz_fitxer fdz, unsigned int cpu,
	unsigned int ptask, unsigned int task, unsigned int thread,
	unsigned long long time, unsigned int com_id, unsigned int send_size,
	unsigned int receive_size, unsigned int glop_id, unsigned int root_rank);

int Paraver_JoinFiles (char *outName, FileSet_t * fset, unsigned long long Ftime,
  int nfiles,  struct Pair_NodeCPU *NodeCPUinfo, int numtasks, int taskid,
  unsigned long long records_per_task);


#endif /* __PARAVER_GENERATOR_H__ */
