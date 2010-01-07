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

#ifndef _LABELS_H
#define _LABELS_H

#include "events.h"

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
#define FLUSH_LBL              "Flushing Traces"
#define READ_LBL               "I/O Read"
#define WRITE_LBL              "I/O Write"

#define TRACING_LBL            "Tracing"

#define TRAC_ENABLED_LBL       "Enabled"
#define TRAC_DISABLED_LBL      "Disabled"

#define IOSIZE_LBL             "I/O Size"

#define MPI_GLOBAL_OP_ROOT_LBL      "Root in MPI Global OP"
#define MPI_GLOBAL_OP_SENDSIZE_LBL  "Send Size in MPI Global OP"
#define MPI_GLOBAL_OP_RECVSIZE_LBL  "Recv Size in MPI Global OP"
#define MPI_GLOBAL_OP_COMM_LBL      "Communicator in MPI Global OP"

#define PACX_GLOBAL_OP_ROOT_LBL     "Root in PACX Global OP"
#define PACX_GLOBAL_OP_SENDSIZE_LBL "Send Size in PACX Global OP"
#define PACX_GLOBAL_OP_RECVSIZE_LBL "Recv Size in PACX Global OP"
#define PACX_GLOBAL_OP_COMM_LBL     "Communicator in PACX Global OP"

#define MISC_GRADIENT   6
#define MISC            4
extern struct evttype_t MISC_events[MISC];
#define MISC_VALUES  2
extern struct value_t MISC_values[MISC_VALUES];

/******************************************************************************
 *   MPI / PACX Caller labels
 ******************************************************************************/

#define CALLER_LBL          "Caller"
#define CALLER_LINE_LBL     "Caller line"
#define CALLER_LVL_LBL      "Caller at level" 
#define CALLER_LINE_LVL_LBL "Caller line at level" 

#define MPI_GRADIENT   1
#define PACX_GRADIENT  1

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

#define STATES_LBL           "STATES"
#define STATES_COLOR_LBL     "STATES_COLOR"

#define STATES_NUMBER        17
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

typedef struct mpi_stats_evt_t
{
	int evt_type;
	char * label;
} mpi_stats_evt_t;

#define MPI_STATS_P2P_COMMS_LBL         "Number of Point-to-Point communications through MPI"
#define MPI_STATS_P2P_BYTES_SENT_LBL    "Point-to-Point bytes sent through MPI"
#define MPI_STATS_P2P_BYTES_RECV_LBL    "Point-to-Point bytes received through MPI"
#define MPI_STATS_GLOBAL_COMMS_LBL      "Number of global operations through MPI"
#define MPI_STATS_GLOBAL_BYTES_SENT_LBL "Global operations bytes sent through MPI"
#define MPI_STATS_GLOBAL_BYTES_RECV_LBL "Global operations bytes received through MPI"
#define MPI_STATS_TIME_IN_MPI_LBL       "Elapsed time in MPI"
extern struct mpi_stats_evt_t mpistats_evt_labels[MPI_STATS_EVENTS_COUNT];

typedef struct pacx_stats_evt_t
{
	int evt_type;
	char * label;
} pacx_stats_evt_t;

#define PACX_STATS_P2P_COMMS_LBL         "Number of Point-to-Point communications through PACX"
#define PACX_STATS_P2P_BYTES_SENT_LBL    "Point-to-Point bytes sent through PACX"
#define PACX_STATS_P2P_BYTES_RECV_LBL    "Point-to-Point bytes received through PACX"
#define PACX_STATS_GLOBAL_COMMS_LBL      "Number of global operations through PACX"
#define PACX_STATS_GLOBAL_BYTES_SENT_LBL "Global operations bytes sent through PACX"
#define PACX_STATS_GLOBAL_BYTES_RECV_LBL "Global operations bytes received through PACX"
#define PACX_STATS_TIME_IN_PACX_LBL      "Elapsed time in PACX"
extern struct pacx_stats_evt_t pacx_stats_evt_labels[PACX_STATS_EVENTS_COUNT];

/* Clustering events labels */
#define CLUSTER_ID_LABEL "Cluster ID"
extern int MaxClusterId;

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

void set_counter_used (long long);
void Address2Info_Write_Labels (FILE *);
int GeneratePCFfile (char *name, long long options);
void loadSYMfile (char *name);

#endif

