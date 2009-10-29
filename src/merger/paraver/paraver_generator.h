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

void trace_paraver_unmatched_communication (unsigned ptask_s,
	unsigned task_s, unsigned thread_s);

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
	unsigned int is_root);

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
