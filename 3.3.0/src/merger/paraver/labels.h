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

#ifndef _LABELS_H
#define _LABELS_H

#include "mpi2out.h"
#include "events.h"
#include <extrae_vector.h>

typedef enum {
	CODELOCATION_FUNCTION,
	CODELOCATION_FILELINE
} codelocation_type_t;

typedef struct codelocation_label_st
{
	int eventcode;
	codelocation_type_t type;
	char *description;
} codelocation_label_t;

/* Codi del David Vicente / PFC */
typedef struct fcounter_t {
	int counter;
	struct fcounter_t *next; 
	struct fcounter_t *prev; 

}fcounter_t;

#define INSERTAR_CONTADOR(fcounter,EvCnt) \
{\
 struct fcounter_t* aux_fc;\
 \
 aux_fc=(struct fcounter_t*) malloc (sizeof(fcounter_t)); \
 if(aux_fc!=NULL) {		\
  aux_fc->counter=EvCnt;\
  aux_fc->prev=fcounter;	\
  if(fcounter!=NULL)  \
   fcounter->next=aux_fc;	\
  fcounter=aux_fc; \
 }\
}

/* Fi codi David Vicente */

#define LET_SPACES(fd) fprintf((fd),"\n\n")

#define TYPE_LBL   150
typedef struct color_t
{
  int value;
  char label[TYPE_LBL];
  int rgb[3];
}
color_t;


#define EVENT_LBL   150
typedef struct evttype_t
{
  int type;
  char label[EVENT_LBL];
}
evttype_t;


#define VALUE_LBL   150
typedef struct value_t
{
  int value;
  char label[VALUE_LBL];
}
value_t;

#define EVT_BEGIN_LBL         "Begin"
#define EVT_END_LBL           "End"

/******************************************************************************
 *   General user events to trace the application.
 ******************************************************************************/

#define APPL_LBL               "Application"
#define TRACE_INIT_LBL         "Trace initialization"
#define FLUSH_LBL              "Flushing Traces"
#define FORK_SYSCALL_LBL       "Process-related syscalls"
#define FORK_LBL               "fork()"
#define WAIT_LBL               "wait()"
#define WAITPID_LBL            "waitpid()"
#define EXEC_LBL               "exec() or similar"
#define SYSTEM_LBL             "system()"
#define GETCPU_LBL             "Executing CPU"
#define PID_LBL                "Process IDentifier"
#define PPID_LBL               "Parent Process IDentifier"
#define FORK_DEPTH_LBL         "fork() depth"
#define LIBRARY_LBL            "Library"

#define TRACING_LBL            "Tracing"

#define TRAC_ENABLED_LBL       "Enabled"
#define TRAC_DISABLED_LBL      "Disabled"

#define MPI_GLOBAL_OP_ROOT_LBL      "Root in MPI Global OP"
#define MPI_GLOBAL_OP_SENDSIZE_LBL  "Send Size in MPI Global OP"
#define MPI_GLOBAL_OP_RECVSIZE_LBL  "Recv Size in MPI Global OP"
#define MPI_GLOBAL_OP_COMM_LBL      "Communicator in MPI Global OP"

#define MISC_GRADIENT   6
#define MISC            4
extern struct evttype_t MISC_events[MISC];
#define MISC_VALUES  2
extern struct value_t MISC_values[MISC_VALUES];

/******************************************************************************
 *   MPI Caller labels
 ******************************************************************************/

#define CALLER_LBL          "Caller"
#define CALLER_LINE_LBL     "Caller line"
#define CALLER_LVL_LBL      "Caller at level" 
#define CALLER_LINE_LVL_LBL "Caller line at level" 


/* Caller of the referenced address */ 
#define SAMPLING_ADDRESS_ALLOCATED_OBJECT_LBL "Memory object referenced by sampled address"

#define MPI_GRADIENT   1

/******************************************************************************
 *   Code Colors.
 ******************************************************************************/

#define STATE_0              0
#define STATE0_LBL           "Idle"
#define STATE0_COLOR         {117,195,255}

#define STATE_1              1
#define STATE1_LBL           "Running"
#define STATE1_COLOR         {0,0,255}

#define STATE_2              2
#define STATE2_LBL           "Not created"
#define STATE2_COLOR         {255,255,255}

#define STATE_3              3
#define STATE3_LBL           "Waiting a message"
#define STATE3_COLOR         {255,0,0}

#define STATE_4              4
#define STATE4_LBL           "Blocking Send"
#define STATE4_COLOR         {255,0,174}

#define STATE_5              5
#define STATE5_LBL           "Synchronization"
#define STATE5_COLOR         {179,0,0}

#define STATE_6              6
#define STATE6_LBL           "Test/Probe"
#define STATE6_COLOR         {0,255,0}

#define STATE_7              7
#define STATE7_LBL           "Scheduling and Fork/Join"
#define STATE7_COLOR         {255,255,0}

#define STATE_8              8
#define STATE8_LBL           "Wait/WaitAll"
#define STATE8_COLOR         {235,0,0}

#define STATE_9              9
#define STATE9_LBL           "Blocked"
#define STATE9_COLOR         {0,162,0}

#define STATE_10             10
#define STATE10_LBL          "Immediate Send"
#define STATE10_COLOR        {255,0,255}

#define STATE_11             11
#define STATE11_LBL          "Immediate Receive"
#define STATE11_COLOR        {100,100,177}

#define STATE_12             12
#define STATE12_LBL          "I/O"
#define STATE12_COLOR        {172,174,41}

#define STATE_13             13
#define STATE13_LBL          "Group Communication"
#define STATE13_COLOR        {255,144,26}

#define STATE_14             14
#define STATE14_LBL          "Tracing Disabled"
#define STATE14_COLOR        {2,255,177}

#define STATE_15             15
#define STATE15_LBL          "Others"
#define STATE15_COLOR        {192,224,0}

#define STATE_16             16
#define STATE16_LBL          "Send Receive"
#define STATE16_COLOR        {66,66,66}

#define STATE_17             17
#define STATE17_LBL          "Memory transfer"
#define STATE17_COLOR        {0xff, 0x0, 0x60}

#define STATE_18             18
#define STATE18_LBL          "Profiling"
#define STATE18_COLOR        {169, 169, 169}

#define STATE_19             19
#define STATE19_LBL          "On-line analysis"
#define STATE19_COLOR        {169, 0, 0}

#define STATE_20             20
#define STATE20_LBL          "Remote memory access"
#define STATE20_COLOR        {   0, 109, 255 }

#define STATE_21             21
#define STATE21_LBL          "Atomic memory operation"
#define STATE21_COLOR        { 200,  61,  68 }

#define STATE_22             22
#define STATE22_LBL          "Memory ordering operation"
#define STATE22_COLOR        { 200,  66,   0 }

#define STATE_23             23
#define STATE23_LBL          "Distributed locking"
#define STATE23_COLOR        {   0,  41,   0 }

#define STATES_LBL           "STATES"
#define STATES_COLOR_LBL     "STATES_COLOR"

#define STATES_NUMBER        24
extern struct color_t states_inf[STATES_NUMBER];

/******************************************************************************
 *   Gradient Colors.
 ******************************************************************************/

#define GRADIENT_0            0
#define GRADIENT0_LBL         "Gradient 0"
#define GRADIENT0_COLOR       {0,255,2}

#define GRADIENT_1            1
#define GRADIENT1_LBL         "Grad. 1/MPI Events"
#define GRADIENT1_COLOR       {0,244,13}

#define GRADIENT_2            2
#define GRADIENT2_LBL         "Grad. 2/OMP Events"
#define GRADIENT2_COLOR       {0,232,25}

#define GRADIENT_3            3
#define GRADIENT3_LBL         "Grad. 3/OMP locks"
#define GRADIENT3_COLOR       {0,220,37}

#define GRADIENT_4            4
#define GRADIENT4_LBL         "Grad. 4/User func"
#define GRADIENT4_COLOR       {0,209,48}

#define GRADIENT_5            5
#define GRADIENT5_LBL         "Grad. 5/User Events"
#define GRADIENT5_COLOR       {0,197,60}

#define GRADIENT_6            6
#define GRADIENT6_LBL         "Grad. 6/General Events"
#define GRADIENT6_COLOR       {0,185,72}

#define GRADIENT_7            7
#define GRADIENT7_LBL         "Grad. 7/Hardware Counters"
#define GRADIENT7_COLOR       {0,173,84}

#define GRADIENT_8            8
#define GRADIENT8_LBL         "Gradient 8"
#define GRADIENT8_COLOR       {0,162,95}

#define GRADIENT_9             9
#define GRADIENT9_LBL         "Gradient 9"
#define GRADIENT9_COLOR       {0,150,107}

#define GRADIENT_10           10
#define GRADIENT10_LBL        "Gradient 10"
#define GRADIENT10_COLOR      {0,138,119}

#define GRADIENT_11           11
#define GRADIENT11_LBL        "Gradient 11"
#define GRADIENT11_COLOR      {0,127,130}

#define GRADIENT_12           12
#define GRADIENT12_LBL        "Gradient 12"
#define GRADIENT12_COLOR      {0,115,142}

#define GRADIENT_13           13
#define GRADIENT13_LBL        "Gradient 13"
#define GRADIENT13_COLOR      {0,103,154}

#define GRADIENT_14           14
#define GRADIENT14_LBL        "Gradient 14"
#define GRADIENT14_COLOR      {0,91,166}

#define GRADIENT_LBL          "GRADIENT_NAMES"
#define GRADIENT_COLOR_LBL    "GRADIENT_COLOR"

#define GRADIENT_NUMBER       15
extern struct color_t gradient_inf[GRADIENT_NUMBER];

typedef struct rusage_evt_t {
	int evt_type;
	char * label;
} rusage_evt_t;

#define RUSAGE_UTIME_LBL    "User time used"
#define RUSAGE_STIME_LBL    "System time used"
#define RUSAGE_MAXRSS_LBL   "Maximum resident set size (in kilobytes)"
#define RUSAGE_IXRSS_LBL    "Text segment memory shared with other processes (kilobyte-seconds)"
#define RUSAGE_IDRSS_LBL    "Data segment memory used (kilobyte-seconds)"
#define RUSAGE_ISRSS_LBL    "Stack memory used (kilobyte-seconds)"
#define RUSAGE_MINFLT_LBL   "Soft page faults"
#define RUSAGE_MAJFLT_LBL   "Hard page faults"
#define RUSAGE_NSWAP_LBL    "Times a process was swapped out of physical memory"
#define RUSAGE_INBLOCK_LBL  "Input operations via the file system"
#define RUSAGE_OUBLOCK_LBL  "Output operations via the file system"
#define RUSAGE_MSGSND_LBL   "IPC messages sent"
#define RUSAGE_MSGRCV_LBL   "IPC messages received"
#define RUSAGE_NSIGNALS_LBL "Signals delivered"
#define RUSAGE_NVCSW_LBL    "Voluntary context switches"
#define RUSAGE_NIVCSW_LBL   "Involuntary context switches"
extern struct rusage_evt_t rusage_evt_labels[RUSAGE_EVENTS_COUNT];

typedef struct memusage_evt_t {
	int evt_type;
	char * label;
} memusage_evt_t;

#define MEMUSAGE_ARENA_LBL		"Total bytes allocated with brk/sbrk"
#define MEMUSAGE_HBLKHD_LBL		"Total bytes allocated with mmap"
#define MEMUSAGE_UORDBLKS_LBL	"Total sbrk memory in use"
#define MEMUSAGE_FORDBLKS_LBL	"Total sbrk memory free"
#define MEMUSAGE_INUSE_LBL		"Total memory in use"
extern struct memusage_evt_t memusage_evt_labels[MEMUSAGE_EVENTS_COUNT];

typedef struct mpi_stats_evt_t
{
	int evt_type;
	char * label;
} mpi_stats_evt_t;

/* Original stats */
#define MPI_STATS_P2P_COUNT_LBL                    "Number of P2P MPI calls"
#define MPI_STATS_P2P_BYTES_SENT_LBL               "Bytes sent in P2P MPI calls"
#define MPI_STATS_P2P_BYTES_RECV_LBL               "Bytes received in P2P MPI calls"
#define MPI_STATS_GLOBAL_COUNT_LBL                 "Number of GLOBAL MPI calls"
#define MPI_STATS_GLOBAL_BYTES_SENT_LBL            "Bytes sent in GLOBAL MPI calls"
#define MPI_STATS_GLOBAL_BYTES_RECV_LBL            "Bytes received in GLOBAL MPI calls"
#define MPI_STATS_TIME_IN_MPI_LBL                  "Elapsed time in MPI"
/* New stats */
#define MPI_STATS_P2P_INCOMING_COUNT_LBL           "Number of incoming P2P MPI calls"
#define MPI_STATS_P2P_OUTGOING_COUNT_LBL           "Number of outgoing P2P MPI calls"
#define MPI_STATS_P2P_INCOMING_PARTNERS_COUNT_LBL  "Number of partners in incoming communications"
#define MPI_STATS_P2P_OUTGOING_PARTNERS_COUNT_LBL  "Number of partners in outgoing communications"
#define MPI_STATS_TIME_IN_OTHER_LBL                "Elapsed time in OTHER MPI calls"
#define MPI_STATS_TIME_IN_P2P_LBL                  "Elapsed time in P2P MPI calls"
#define MPI_STATS_TIME_IN_GLOBAL_LBL               "Elapsed time in GLOBAL MPI calls"
#define MPI_STATS_OTHER_COUNT_LBL                  "Number of OTHER MPI calls"

extern struct mpi_stats_evt_t mpistats_evt_labels[MPI_STATS_EVENTS_COUNT];

/* Clustering events labels */
#define CLUSTER_ID_LABEL   "Cluster ID"
extern unsigned int MaxClusterId;

#define PERIODICITY_LABEL     "Representative periods"
#define DETAIL_LEVEL_LABEL    "Detail level"
#define RAW_PERIODICITY_LABEL "Raw periodic zone"
#define RAW_BEST_ITERS_LABEL  "Raw best iterations"
extern unsigned int MaxRepresentativePeriod;
extern unsigned int HaveSpectralEvents;

#define TYPE_LABEL           "EVENT_TYPE"
#define VALUES_LABEL         "VALUES"

/*
 * Default Paraver Options
 */

#define DEFAULT_LEVEL               "THREAD"
#define DEFAULT_SPEED               1
#define DEFAULT_UNITS               "NANOSEC"
#define DEFAULT_LOOK_BACK           100
#define DEFAULT_FLAG_ICONS          "ENABLED"
#define DEFAULT_NUM_OF_STATE_COLORS 1000
#define DEFAULT_YMAX_SCALE          37

#define DEFAULT_THREAD_FUNC    "State As Is"

#define LABELS_ERROR(x) \
   if ( (x) < 0 ) {\
        fprintf(stderr,"ERROR : Writing to disk the tracefile\n"); \
        return (-1); \
   }

void Address2Info_Write_Labels (FILE *);
int Labels_GeneratePCFfile (char *name, long long options);
void Labels_loadSYMfile (int taskid, int allobjects, unsigned ptask,
	unsigned task, char *name, int report);
void Labels_loadLocalSymbols (int taskid, unsigned long nfiles,
	struct input_t * IFiles);
int Labels_LookForHWCCounter (int eventcode, unsigned *position, char **description);

#endif

