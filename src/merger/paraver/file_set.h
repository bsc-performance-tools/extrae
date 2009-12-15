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

#ifndef _FILE_SET_H
#define _FILE_SET_H

#include <config.h>

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "common.h"
#include "queue.h"
#include "new-queue.h"
#include "send_queue.h"
#include "recv_queue.h"
#include "mpi2out.h"
#include "write_file_buffer.h"

#define MAX_FILES 16*1024

enum
{
    CIRCULAR_SKIP_EVENTS,
    CIRCULAR_SKIP_MATCHES,
};

typedef enum
{
	LOGICAL_COMMUNICATION = 0,
	PHYSICAL_COMMUNICATION,
	NUM_COMMUNICATION_TYPE
}
CommunicationType;

typedef enum {
	UNFINISHED_STATE = -1,
	STATE = 1,
	EVENT = 2,
	UNMATCHED_COMMUNICATION = -3,
	COMMUNICATION = 3,
	PENDING_COMMUNICATION = -4
}
RecordType;

typedef struct
{
	unsigned long long receive[NUM_COMMUNICATION_TYPE];
	                              /* both logical and physical recv times */
	UINT64 value;                 /* state, value, tag or global_op_id */
	unsigned long long time;      /* state ini_time or logical send time */
	unsigned long long end_time;  /* state end_time or physical send time */
	RecordType type;              /* Type of record */
	int event;                    /* size, type or comm_id */
	int cpu, ptask, task, thread; /* ID of this task */
	int cpu_r, ptask_r, task_r, thread_r;
	                              /* ID of the receiver task */
}
paraver_rec_t;

typedef struct
{
	int fd;
	/* int output_fd; */
	WriteFileBuffer_t *wfb;
	unsigned long size;
	unsigned int cpu, ptask, task, thread;
	unsigned long long num_of_events;

	event_t *current;
	event_t *next_cpu_burst;
	event_t *first, *last, *first_glop;
	event_t *last_recv;
	event_t *tmp;
	NewQueue_t *recv_queue;
	NewQueue_t *send_queue;
}
FileItem_t;

typedef struct
{
	FileItem_t files[MAX_FILES];    /* Files in the set */
	unsigned int nfiles;            /* Number of files in the set */
	unsigned int traceformat;       /* Output trace format */
	unsigned int active_file;       /* Dimemas uses this in order to know which file is being translated */
	FILE *output_file;              /* Dimemas output file */
	struct input_t *input_files;    /* Input files */
	unsigned int num_input_files;   /* Num of input files */
}
FileSet_t;

typedef enum
{
	LOCAL,
	REMOTE
}
FileItemType;

typedef struct
{
	paraver_rec_t *current_p, *first_mapped_p, *last_mapped_p;
	long long remaining_records;
	long long mapped_records;
	unsigned source;
	FileItemType type;
}
PRVFileItem_t;

typedef struct
{
	PRVFileItem_t files[MAX_FILES];
	unsigned long long records_per_block;
	unsigned int nfiles;
	FileSet_t *fset;
}
PRVFileSet_t;

#define CurrentObj_FS(fitem,fcpu,fptask,ftask,fthread) \
	((fptask) = (fitem)->ptask, (ftask) = (fitem)->task, (fthread) = (fitem)->thread, (fcpu) = (fitem)->cpu)

#define Current_FS(fitem) \
	(((fitem)->current >= (fitem)->last) ? NULL : (fitem)->current)

#define StepOne_FS(fitem) \
	((fitem)->current++)

#define Recv_Queue_FS(fitem) \
	(&((fitem)->recv_queue))

#define Send_Queue_FS(fitem) \
	(&((fitem)->send_queue))

#define NextRecv_FS(fitem)  \
	(((fitem)->last_recv == (fitem)->last) ? NULL : (fitem)->last_recv++)

#define NextRecvG_FS(fitem)  \
	(((fitem)->tmp == (fitem)->last) ? NULL : (fitem)->tmp++)

#ifdef HAVE_LIMITS_H
# include <limits.h>
#endif
#ifdef HAVE_LINUX_LIMITS_H
# include <linux/limits.h>
#endif

FileSet_t *Create_FS (unsigned long nfiles, struct input_t * IFiles, int idtask, int trace_format);
void Free_FS (FileSet_t *fset);
void Flush_FS (FileSet_t *fset, int remove_last);

unsigned int GetActiveFile (FileSet_t *fset);
event_t *GetNextEvent_FS (FileSet_t * fset, unsigned int *cpu, unsigned int *ptask,
	unsigned int *task,  unsigned int *thread);
void Rewind_FS (FileSet_t * fs);
unsigned long long EventsInFS (FileSet_t * fs);

int SearchRecvEvent_FS (FileSet_t *fset, unsigned int ptask,
	unsigned int receiver, unsigned int sender, unsigned int tag,
	event_t ** recv_begin, event_t ** recv_end);
int SearchSendEvent_FS (FileSet_t *fset, unsigned int ptask,
	unsigned int sender, unsigned int receiver, unsigned int tag,
	event_t ** send_begin, event_t ** send_end);
event_t *SearchIRECVED (event_t * current, long long request, FileItem_t * freceive);

long long GetTraceOptions (FileSet_t * fset, int numtasks, int taskid);
int Search_Synchronization_Times (FileSet_t * fset, UINT64 **io_StartingTimes, UINT64 **io_SynchronizationTimes);
void CheckCircularBufferWhenTracing (FileSet_t * fset, int numtasks, int taskid);
int CheckBursts (FileSet_t *fset, int numtasks, int idtask);
void setLimitOfEvents (int limit);
int getBehaviourForCircularBuffer (void);
int tracingCircularBuffer (void);
int getTagForCircularBuffer (void);
void MatchComms_On(unsigned int ptask, unsigned int task, unsigned int thread);
void MatchComms_Off(unsigned int ptask, unsigned int task, unsigned int thread);
int MatchComms_Enabled(unsigned int ptask, unsigned int task, unsigned int thread);
int num_Files_FS (FileSet_t * fset);
void GetNextObj_FS (FileSet_t * fset, int file, unsigned int *cpu,
	unsigned int *ptask, unsigned int *task, unsigned int *thread);

#if defined(HETEROGENEOUS_SUPPORT)
void EndianCorrection (FileSet_t *fset, int numtasks, int taskid);
#endif

int isTaskInMyGroup (FileSet_t *fset, int task);
int inWhichGroup (int task, FileSet_t *fset);

PRVFileSet_t * Map_Paraver_files (FileSet_t * fset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long events_per_block);

paraver_rec_t *GetNextParaver_Rec (PRVFileSet_t * fset, int taskid);

#endif
