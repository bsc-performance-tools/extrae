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

#ifndef _FILE_SET_H
#define _FILE_SET_H

#include <config.h>

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "record.h"
#include "common.h"
#include "queue.h"
#include "mpi2out.h"
#include "write_file_buffer.h"

enum
{
    CIRCULAR_SKIP_EVENTS,
    CIRCULAR_SKIP_MATCHES
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
	unsigned mpit_id;

	event_t *current;
	event_t *next_cpu_burst;
	event_t *first, *last, *first_glop;
	event_t *last_recv;
	event_t *tmp;
}
FileItem_t;

typedef struct
{
	FileItem_t  *files;             /* Files in the set */
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
	WriteFileBuffer_t *destination;
	long long remaining_records;
	long long mapped_records;
	unsigned source;
	FileItemType type;
}
PRVFileItem_t;

typedef struct
{
	PRVFileItem_t *files;
	unsigned long long records_per_block;
	unsigned int nfiles;
	FileSet_t *fset;
	int SkipAsMasterOfSubtree;
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
event_t *Search_MPI_IRECVED (event_t * current, long long request, FileItem_t * freceive);

long long GetTraceOptions (FileSet_t * fset, int numtasks, int taskid);
int Search_Synchronization_Times (int taskid, int ntasks, FileSet_t * fset, UINT64 **io_StartingTimes, UINT64 **io_SynchronizationTimes);
void CheckCircularBufferWhenTracing (FileSet_t * fset, int numtasks, int taskid);
int CheckBursts (FileSet_t *fset, int numtasks, int idtask);
void setLimitOfEvents (int limit);
int getBehaviourForCircularBuffer (void);
int tracingCircularBuffer (void);
int getTagForCircularBuffer (void);
void MatchComms_On(unsigned int ptask, unsigned int task);
void MatchComms_Off(unsigned int ptask, unsigned int task);
void MatchComms_ChangeZone(unsigned int ptask, unsigned int task);
int  MatchComms_GetZone(unsigned int ptask, unsigned int task);
int MatchComms_Enabled(unsigned int ptask, unsigned int task);
int num_Files_FS (FileSet_t * fset);
void GetNextObj_FS (FileSet_t * fset, int file, unsigned int *cpu,
	unsigned int *ptask, unsigned int *task, unsigned int *thread);

#if defined(HETEROGENEOUS_SUPPORT)
void EndianCorrection (FileSet_t *fset, int numtasks, int taskid);
#endif

int isTaskInMyGroup (FileSet_t *fset, int ptask, int task);
int inWhichGroup (int ptask, int task, FileSet_t *fset);

#if defined(PARALLEL_MERGE)

PRVFileSet_t * Map_Paraver_files (FileSet_t * fset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long records_per_block, int fan_out);
PRVFileSet_t * ReMap_Paraver_files_binary (PRVFileSet_t * fset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long records_per_block, int depth, int fan_out);

void Free_Map_Paraver_Files (PRVFileSet_t * infset);

void Flush_Paraver_Files_binary (PRVFileSet_t *prvfset, int taskid, int depth,
	int tree_fan_out);

#else

PRVFileSet_t * Map_Paraver_files (FileSet_t * fset, 
	unsigned long long *num_of_events, int numtasks, int taskid, 
	unsigned long long events_per_block);

#endif /* PARALLEL_MERGE */

paraver_rec_t *GetNextParaver_Rec (PRVFileSet_t * fset);

#endif
