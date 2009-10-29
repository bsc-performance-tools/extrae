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

#ifndef PARALLEL_MERGE_AUX_H
#define PARALLEL_MERGE_AUX_H

#include <config.h>

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "common.h"
#include "file_set.h"

struct Communicator_t
{
	int *tasks;
	int type;
	int task;
	int ptask;
	int id;
	int ntasks;
};
struct Communicators_t
{
	struct Communicator_t *comms;
	int count, size;
};
struct Communicators_t Communicators;

struct PendingCommunication_t
{
	int sender, recver, tag, descriptor, match;
	off_t offset;
};
struct PendingComms_t 
{
	struct PendingCommunication_t *data;
	int count, size;
};
struct PendingComms_t PendingComms;

struct ForeignRecv_t
{
	UINT64 physic, logic;
	int sender, recver, tag;
};
struct ForeignRecvs_t
{
	int count, size;
	struct ForeignRecv_t *data;
};
struct ForeignRecvs_t *ForeignRecvs;

void InitForeignRecvs (int numtasks);
void AddForeignRecv (UINT64 physic, UINT64 logic, int tag, int task_r, 
	int task_s, FileSet_t *fset);

void DistributePendingComms (int numtasks, int taskid);
void NewDistributePendingComms (int numtasks, int taskid, int match);
struct ForeignRecv_t* SearchForeignRecv (int group, int sender, int recver, int tag);
void AddPendingCommunication (int descriptor, off_t offset, int tag, int task_r,
	int task_s);
void InitPendingCommunication (void);

void InitCommunicators(void);
void AddCommunicator (int ptask, int task, int type, int id, int ntasks, int *tasks);
void BuildCommunicators (int num_tasks, int taskid);

void ShareNodeNames (int numtasks, char ***nodenames);

void ShareTraceInformation (int numtasks, int taskid);

void Gather_Dimemas_Traces (int numtasks, int taskid, FILE *fd, unsigned int maxmem);
void Gather_Dimemas_Offsets (int numtasks, int taskid, int count,
	unsigned long long *in_offsets, unsigned long long **out_offsets,
	unsigned long long local_trace_size, FileSet_t *fset);

#endif /* PARALLEL_MERGE_AUX_H */
