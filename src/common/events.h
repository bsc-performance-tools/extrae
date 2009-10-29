/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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

#ifndef __EVENTS_H_INCLUDED__
#define __EVENTS_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif
unsigned IsMPI (unsigned EvType);
unsigned IsOpenMP (unsigned EvType);
unsigned IsMISC (unsigned EvType);
unsigned IsTRT (unsigned EvType);

unsigned IsBurst (unsigned EvType);
unsigned IsHwcChange (unsigned EvType);
unsigned IsMPICollective (unsigned EvType);
#ifdef __cplusplus
}
#endif


#define EMPTY            ( 0)
#define NO_COUNTER       (-1)
#define SAMPLE_COUNTER   (-2)

#define MAX_CALLERS      100

/******************************************************************************
 *   General user events to trace the application.
 ******************************************************************************/

#define NULL_EV -1

/* Trace options, just do a bitwise or/and with these values */
#define TRACEOPTION_NONE            (0)
#define TRACEOPTION_HWC             (1<<0)
#define TRACEOPTION_CIRCULAR_BUFFER (1<<1)
/* Useless #define TRACEOPTION_BURSTS          (1<<2) */
#define TRACEOPTION_BIGENDIAN       (1<<3)
#define TRACEOPTION_PARAVER         (1<<4)
#define TRACEOPTION_DIMEMAS         (1<<5)

#define SYNCHRONIZATION_POINT_EV     1000
#define OPTIONS_EV                   1001

#define SAMPLING_EV              30000000
#define SAMPLING_LINE_EV         30000100
#define SAMPLING_CALLER_OFFSET     100000
#define HWC_SET_OVERFLOW_EV      31000000

#define APPL_EV                  40000001
#define FLUSH_EV                 40000003
#define READ_EV                  40000004
#define WRITE_EV                 40000005
#define USER_EV                  40000006
#define HWC_DEF_EV				 40000007
#define HWC_CHANGE_EV            40000008
#define HWC_EV                   40000009
#define IOSIZE_EV                40000011
#define TRACING_EV               40000012
#define SET_TRACE_EV             40000014
#define CPU_BURST_EV             40000015
#define RUSAGE_EV                40000016
#define MPI_STATS_EV             40000017
#define TRACING_MODE_EV          40000018

#define RUSAGE_BASE              45000000
enum {
   RUSAGE_UTIME_EV = 0,
   RUSAGE_STIME_EV,
   RUSAGE_MAXRSS_EV,
   RUSAGE_IXRSS_EV,
   RUSAGE_IDRSS_EV,
   RUSAGE_ISRSS_EV,
   RUSAGE_MINFLT_EV,
   RUSAGE_MAJFLT_EV,
   RUSAGE_NSWAP_EV,
   RUSAGE_INBLOCK_EV,
   RUSAGE_OUBLOCK_EV,
   RUSAGE_MSGSND_EV,
   RUSAGE_MSGRCV_EV,
   RUSAGE_NSIGNALS_EV,
   RUSAGE_NVCSW_EV,
   RUSAGE_NIVCSW_EV,
   RUSAGE_EVENTS_COUNT /* Total number of getrusage events */
};

#define MPI_STATS_BASE           54000000
enum {
   MPI_STATS_P2P_COMMS_EV = 0,
   MPI_STATS_P2P_BYTES_SENT_EV,
   MPI_STATS_P2P_BYTES_RECV_EV,
   MPI_STATS_GLOBAL_COMMS_EV,
   MPI_STATS_GLOBAL_BYTES_SENT_EV,
   MPI_STATS_GLOBAL_BYTES_RECV_EV,
   MPI_STATS_TIME_IN_MPI_EV, 
   MPI_STATS_EVENTS_COUNT /* Total number of MPI statistics */
};

#define FUNCT_BASE               41000000
#define FUNCT_MAX                    1000

#define HWC_BASE                 42000000 /* Per als comptadors base de PAPI */
#define HWC_BASE_NATIVE          42001000 /* Per als comptadors natius de PAPI 
                                             Cal comprovar que poguem codificar
                                             tots els comptadors en els 3 zeros
                                             que hi ha disponible! */
#define HWC_GROUP_ID             42009999 /* Per indicar l'identificador de grup */

/******************************************************************************
 *   User events to trace several MPI functions.
 *   MUST be between 50000001 - 59999999
 ******************************************************************************/
#define MPI_MIN_EV               MPIINIT_EV
#define MPI_MAX_EV               59999999

#define MPIINIT_EV               50000001
#define BSEND_EV                 50000002
#define SSEND_EV                 50000003
#define BARRIER_EV               50000004
#define BCAST_EV                 50000005
#define SEND_EV                  50000018
#define SENDRECV_EV              50000017
#define SENDRECV_REPLACE_EV      50000081
#define RECV_EV                  50000019
#define IBSEND_EV                50000020
#define ISSEND_EV                50000021
#define ISEND_EV                 50000022
#define IRECV_EV                 50000023
#define ISND_EV                  50000024       /* Not used */
#define IRCV_EV                  50000025
#define TEST_EV                  50000026
#define TEST_COUNTER_EV          50000080
#define WAIT_EV                  50000027
#define TESTFOR_EV               50000028       /* Not used */
#define ENDICOMM_EV              50000029       /* Not used */
#define CANCEL_EV                50000030
#define RSEND_EV                 50000031
#define IRSEND_EV                50000032
#define ALLTOALL_EV              50000033
#define ALLTOALLV_EV             50000034
#define ALLREDUCE_EV             50000035
#define REDUCE_EV                50000038
#define WAITALL_EV               50000039
#define WAITANY_EV               50000068
#define WAITSOME_EV              50000069
#define IRECVED_EV               50000040
#define GATHER_EV                50000041
#define GATHERV_EV               50000042
#define SCATTER_EV               50000043
#define SCATTERV_EV              50000044
#define FINALIZE_EV              50000045
#define COMM_RANK_EV             50000046
#define COMM_SIZE_EV             50000047
#define COMM_CREATE_EV           50000048
#define COMM_DUP_EV              50000049
#define COMM_SPLIT_EV            50000050
#define RANK_CREACIO_COMM_EV     50000051      /* Used to define communicators */
#define ALLGATHER_EV             50000052
#define ALLGATHERV_EV            50000053
#define MPI_CART_CREATE_EV       50000058
#define MPI_CART_SUB_EV          50000059
#define MPI_CART_COORDS_EV       50000060
#define REDUCESCAT_EV            50000062
#define SCAN_EV                  50000063
#define PROBE_EV                 50000065
#define IPROBE_EV                50000066

#define PERSIST_REQ_EV           50000070
#define START_EV                 50000071
#define STARTALL_EV              50000072
#define REQUEST_FREE_EV          50000073
#define RECV_INIT_EV             50000074
#define SEND_INIT_EV             50000075
#define BSEND_INIT_EV            50000076
#define RSEND_INIT_EV            50000077
#define SSEND_INIT_EV            50000078

#define DIMEMAS_DEF_EV           50000054      /* Definicio d'un evant d'usuari */
#define DIMEMAS_EVENT_EV         50000055      /* Event d'usuari */
#define DIMEMAS_STRING_EV        50000056      /* Un string (o una part) en forma d'event */
#define DIMEMAS_TRACE_EV         50000057      /* Inici/Fi del traceig */

#define IPROBE_COUNTER_EV        53000000
#define TIME_OUTSIDE_IPROBES_EV  53000001

#define MPI_GLOBAL_OP_SENDSIZE   (MPIINIT_EV+1000000)
#define MPI_GLOBAL_OP_RECVSIZE   (MPIINIT_EV+1000001)
#define MPI_GLOBAL_OP_ROOT       (MPIINIT_EV+1000002)
#define MPI_GLOBAL_OP_COMM       (MPIINIT_EV+1000003)

#define FILE_OPEN_EV             50000100
#define FILE_CLOSE_EV            50000101
#define FILE_READ_EV             50000102
#define FILE_READ_ALL_EV         50000103
#define FILE_WRITE_EV            50000104 
#define FILE_WRITE_ALL_EV        50000105
#define FILE_READ_AT_EV          50000106
#define FILE_READ_AT_ALL_EV      50000107
#define FILE_WRITE_AT_EV         50000108
#define FILE_WRITE_AT_ALL_EV     50000109

/******************************************************************************
 *   User events for BG PERSONALITY
 ******************************************************************************/
#if defined(IS_BG_MACHINE)
#define BGL_PERSONALITY_TORUS_X  6000
#define BGL_PERSONALITY_TORUS_Y  6001
#define BGL_PERSONALITY_TORUS_Z  6002
#define BGL_PERSONALITY_PROCESSOR_ID 6003
#endif

/******************************************************************************
 *   User events to trace MN topology (grodrigu)
 ******************************************************************************/

#if defined(IS_MN_MACHINE)
# define LINEAR_HOST_EVENT       3000
# define LINECARD_EVENT          3001
# define HOST_EVENT              3002
#endif

/******************************************************************************
 *   User events to trace OMP parallel execution.
 ******************************************************************************/
#define PAR_EV                   60000001
#define WSH_EV                   60000002
#define BLOCK_EV                 60000003
#define WWORK_EV                 60000004
#define BARRIEROMP_EV            60000005
#define NAMEDCRIT_EV             60000006
#define UNNAMEDCRIT_EV           60000007
#define INTLOCK_EV               60000008
#define OMPLOCK_EV               60000009
#define OVHD_EV                  60000010
#define WORK_EV                  60000011
#define ENTERGATE_EV             60000012
#define EXITGATE_EV              60000013
#define ORDBEGIN_EV              60000014
#define ORDEND_EV                60000015
#define JOIN_EV                  60000016
#define DESCMARK_EV              60000017
#define OMPFUNC_EV               60000018
#define OMPFUNC_LINE_EV          60000118
#define USRFUNC_EV               60000019
#define USRFUNC_LINE_EV          60000119

/******************************************************************************
 *   User events to trace Pthread/TRT parallel execution.
 ******************************************************************************/
#define PTHREADCREATE_EV         61000001
#define PTHREADJOIN_EV           61000002
#define PTHREADDETACH_EV         61000003
#define PTHREADFUNC_EV           60000020
#define PTHREADFUNC_LINE_EV      60000120

#define TRT_SPAWN_EV             62000001
#define TRT_READ_EV              62000002
#define TRT_USRFUNC_EV           62000003
#define TRT_USRFUNC_LINE_EV      62000003

#define MPI_CALLER_EV            70000000
#define MPI_CALLER_LINE_EV       80000000

#define MRNET_EV                 50000
#define CLUSTER_ID_EV            90000001

/* 
 * Values.
 */

#define WORK_WSH_VAL             1
#define WORK_REG_VAL             2
#define WORK_DOSINGLE_VAL        3 /* The thread goes to do the single section */

/* 
 * Parallelism values.
 */
#define PAR_END_VAL              0 /* Close parallel (region and * worksharing constructs). */
#define PAR_WSH_VAL              1 /* Parallel worksharing constructs : * PARALLEL DO */
#define PAR_SEC_VAL              2 /* Parallel worksharing constructs : * PARALLEL SECTIONS */
#define PAR_REG_VAL              3 /* Parallel region construct : * PARALLEL. */
/* 
 * Worksharing construct values
 */
#define WSH_END_VAL              0 /* worsharing ending : DO, SINGLE * and SECTIONS */
#define WSH_DO_VAL               4 /* worksharing constructs : DO * and SECTIONS. */
#define WSH_SEC_VAL              5 /* worksharing constructs : DO * and SECTIONS. */
#define WSH_SINGLE_VAL           6 /* worksharing construct : SINGLE */ 

/* Workharing ending values */
#define JOIN_WAIT_VAL            1
#define JOIN_NOWAIT_VAL          2

/* 
 * Lock Values.
 */
#define UNLOCKED_VAL             0 /* Unlocked Status. Mutex is unlocked. */
#define LOCK_VAL                 3 /* Inside an acquire lock function. */
#define UNLOCK_VAL               5 /* Inside a release lock function. */
#define LOCKED_VAL               6 /* Locked Status. Mutex is locked. */

/* 
 * Some Ordered Values.
 */
#define IT_MARK_VAL             2
#define WAIT_BEGIN_VAL          3
#define WAIT_END_VAL            4

#if defined(DEAD_CODE)

/*************************************************************************
 * Base identifier for locks and functions.
 *************************************************************************/
#define CRTLOCK_BASE   61000000 /* Critical sections lock base type. */
#   define CRTLOCK_MAX     1000 /* Maximum number of locks. */

/* Adress lock used by omp_init_lock, omp_set_lock, omp_unset_lock */
#define OMPLOCK_BASE   62000000 /* OMP lock base type. */
#   define OMPLOCK_MAX     1000 /* Maximum number of locks. */

#define OMPFUNC_BASE          1 /* OMP functions base type. */
#   define OMPFUNC_MAX     1000 /* Maximum number of functions. */

#endif /* DEAD_CODE */


/* Values */
#define EVT_BEGIN                1
#define EVT_END                  0

#if defined(DEAD_CODE)
# define EVT_REND                2
#endif /* DEAD_CODE */

#if defined(DEAD_CODE)
/* 
 * Base value for reduce operation event value where
 * process in a NONROOT.
 */
#define BASE_OP_VALUE            50000000
#endif /* DEAD_CODE */

/* Root/Nonroot process in a Reduce opration */
#define ROOT                     1
#define NONROOT                  0

#define STATE_ANY				-1
#define STATE_IDLE               0
#define STATE_RUNNING            1
#define STATE_STOPPED            2
#define STATE_WAITMESS           3
#define STATE_BLOCKED            9
#define STATE_SYNC               5
#define STATE_BARRIER            5
#define STATE_TWRECV             8
#define STATE_OVHD               7
#define STATE_PROBE              6
#define STATE_BSEND              4
#define STATE_SEND               4
#define STATE_RSEND              4
#define STATE_SSEND              4
#define STATE_IBSEND             10
#define STATE_ISEND              10
#define STATE_IRSEND             10
#define STATE_ISSEND             10
#define STATE_IWAITMESS          11
#define STATE_IRECV              11
#define STATE_IO                 12
#define STATE_FLUSH              12
#define STATE_BCAST              13
#define STATE_NOT_TRACING        14
#define STATE_INITFINI           15
#define STATE_MIXED              15
#define STATE_SENDRECVOP         16

#if defined(DEAD_CODE)
/* ==========================================================================
   ==== MPI Dimemas Block Numbers
   ========================================================================== */

typedef enum
{
/* 000 */   BLOCK_ID_NULL,
/* 001 */   BLOCK_ID_MPI_Allgather,
/* 002 */   BLOCK_ID_MPI_Allgatherv,
/* 003 */   BLOCK_ID_MPI_Allreduce,
/* 004 */   BLOCK_ID_MPI_Alltoall,
/* 005 */   BLOCK_ID_MPI_Alltoallv,
/* 006 */   BLOCK_ID_MPI_Barrier,
/* 007 */   BLOCK_ID_MPI_Bcast,
/* 008 */   BLOCK_ID_MPI_Gather,
/* 009 */   BLOCK_ID_MPI_Gatherv,
/* 010 */   BLOCK_ID_MPI_Op_create,
/* 011 */   BLOCK_ID_MPI_Op_free,
/* 012 */   BLOCK_ID_MPI_Reduce_scatter,
/* 013 */   BLOCK_ID_MPI_Reduce,
/* 014 */   BLOCK_ID_MPI_Scan,
/* 015 */   BLOCK_ID_MPI_Scatter,
/* 016 */   BLOCK_ID_MPI_Scatterv,
/* 017 */   BLOCK_ID_MPI_Attr_delete,
/* 018 */   BLOCK_ID_MPI_Attr_get,
/* 019 */   BLOCK_ID_MPI_Attr_put,
  
/* 020 */   BLOCK_ID_MPI_Comm_create,
/* 021 */   BLOCK_ID_MPI_Comm_dup,
/* 022 */   BLOCK_ID_MPI_Comm_free,
/* 023 */   BLOCK_ID_MPI_Comm_group,
/* 024 */   BLOCK_ID_MPI_Comm_rank,
/* 025 */   BLOCK_ID_MPI_Comm_remote_group,
/* 026 */   BLOCK_ID_MPI_Comm_remote_size,
/* 027 */   BLOCK_ID_MPI_Comm_size,
/* 028 */   BLOCK_ID_MPI_Comm_split,
/* 029 */   BLOCK_ID_MPI_Comm_test_inter,
/* 030 */   BLOCK_ID_MPI_Comm_compare,
/* 031 */   BLOCK_ID_MPI_Group_difference,
/* 032 */   BLOCK_ID_MPI_Group_excl,
/* 033 */   BLOCK_ID_MPI_Group_free,
/* 034 */   BLOCK_ID_MPI_Group_incl,
/* 035 */   BLOCK_ID_MPI_Group_intersection,
/* 036 */   BLOCK_ID_MPI_Group_rank,
/* 037 */   BLOCK_ID_MPI_Group_range_excl,
/* 038 */   BLOCK_ID_MPI_Group_range_incl,
/* 039 */   BLOCK_ID_MPI_Group_size,
/* 040 */   BLOCK_ID_MPI_Group_translate_ranks,
/* 041 */   BLOCK_ID_MPI_Group_union,
/* 042 */   BLOCK_ID_MPI_Group_compare,
/* 043 */   BLOCK_ID_MPI_Intercomm_create,
/* 044 */   BLOCK_ID_MPI_Intercomm_merge,
/* 045 */   BLOCK_ID_MPI_Keyval_free,
/* 046 */   BLOCK_ID_MPI_Keyval_create,
/* 047 */   BLOCK_ID_MPI_Abort,
/* 048 */   BLOCK_ID_MPI_Error_class,
/* 049 */   BLOCK_ID_MPI_Errhandler_create,
/* 050 */   BLOCK_ID_MPI_Errhandler_free,
/* 051 */   BLOCK_ID_MPI_Errhandler_get,
/* 052 */   BLOCK_ID_MPI_Error_string,
/* 053 */   BLOCK_ID_MPI_Errhandler_set,
/* 054 */   BLOCK_ID_MPI_Finalize,
/* 055 */   BLOCK_ID_MPI_Get_processor_name,
/* 056 */   BLOCK_ID_MPI_Init,
/* 057 */   BLOCK_ID_MPI_Initialized,
/* 058 */   BLOCK_ID_MPI_Wtick,
/* 059 */   BLOCK_ID_MPI_Wtime,
/* 060 */   BLOCK_ID_MPI_Address,
/* 061 */   BLOCK_ID_MPI_Bsend,
/* 062 */   BLOCK_ID_MPI_Bsend_init,
/* 063 */   BLOCK_ID_MPI_Buffer_attach,
/* 064 */   BLOCK_ID_MPI_Buffer_detach,
/* 065 */   BLOCK_ID_MPI_Cancel,
/* 066 */   BLOCK_ID_MPI_Request_free,
/* 067 */   BLOCK_ID_MPI_Recv_init,
/* 068 */   BLOCK_ID_MPI_Send_init,
/* 069 */   BLOCK_ID_MPI_Get_count,
/* 070 */   BLOCK_ID_MPI_Get_elements,
/* 071 */   BLOCK_ID_MPI_Ibsend,
/* 072 */   BLOCK_ID_MPI_Iprobe,
/* 073 */   BLOCK_ID_MPI_Irecv,
/* 074 */   BLOCK_ID_MPI_Irsend,
/* 075 */   BLOCK_ID_MPI_Isend,
/* 076 */   BLOCK_ID_MPI_Issend,
/* 077 */   BLOCK_ID_MPI_Pack,
/* 078 */   BLOCK_ID_MPI_Pack_size,
/* 079 */   BLOCK_ID_MPI_Probe,
/* 080 */   BLOCK_ID_MPI_Recv,
/* 081 */   BLOCK_ID_MPI_Rsend,
/* 082 */   BLOCK_ID_MPI_Rsend_init,
/* 083 */   BLOCK_ID_MPI_Send,
/* 084 */   BLOCK_ID_MPI_Sendrecv,
/* 085 */   BLOCK_ID_MPI_Sendrecv_replace,
/* 086 */   BLOCK_ID_MPI_Ssend,
/* 087 */   BLOCK_ID_MPI_Ssend_init,
/* 088 */   BLOCK_ID_MPI_Start,
/* 089 */   BLOCK_ID_MPI_Startall,
/* 090 */   BLOCK_ID_MPI_Test,
/* 091 */   BLOCK_ID_MPI_Testall,
/* 092 */   BLOCK_ID_MPI_Testany,
/* 093 */   BLOCK_ID_MPI_Test_cancelled,
/* 094 */   BLOCK_ID_MPI_Test_some,
/* 095 */   BLOCK_ID_MPI_Type_commit,
/* 096 */   BLOCK_ID_MPI_Type_contiguous,
/* 097 */   BLOCK_ID_MPI_Type_extent,
/* 098 */   BLOCK_ID_MPI_Type_free,
/* 099 */   BLOCK_ID_MPI_Type_hindexed,
/* 100 */   BLOCK_ID_MPI_Type_hvector,
/* 101 */   BLOCK_ID_MPI_Type_indexed,
/* 102 */   BLOCK_ID_MPI_Type_lb,
/* 103 */   BLOCK_ID_MPI_Type_size,
/* 104 */   BLOCK_ID_MPI_Type_struct,
/* 105 */   BLOCK_ID_MPI_Type_ub,
/* 106 */   BLOCK_ID_MPI_Type_vector,
/* 107 */   BLOCK_ID_MPI_Unpack,
/* 108 */   BLOCK_ID_MPI_Wait,
/* 109 */   BLOCK_ID_MPI_Waitall,
/* 110 */   BLOCK_ID_MPI_Waitany,
/* 111 */   BLOCK_ID_MPI_Waitsome,
/* 112 */   BLOCK_ID_MPI_Cart_coords,
/* 113 */   BLOCK_ID_MPI_Cart_create,
/* 114 */   BLOCK_ID_MPI_Cart_get,
/* 115 */   BLOCK_ID_MPI_Cart_map,
/* 116 */   BLOCK_ID_MPI_Cart_rank,
/* 117 */   BLOCK_ID_MPI_Cart_shift,
/* 118 */   BLOCK_ID_MPI_Cart_sub,
/* 119 */   BLOCK_ID_MPI_Cartdim_get,
/* 120 */   BLOCK_ID_MPI_Dims_create,
/* 121 */   BLOCK_ID_MPI_Graph_get,
/* 122 */   BLOCK_ID_MPI_Graph_map,
/* 123 */   BLOCK_ID_MPI_Graph_create,
/* 124 */   BLOCK_ID_MPI_Graph_neighbors,
/* 125 */   BLOCK_ID_MPI_Graphdims_get,
/* 126 */   BLOCK_ID_MPI_Graph_neighbors_count,
/* 127 */   BLOCK_ID_MPI_Topo_test,

/* 128 */   BLOCK_ID_TRACE_ON,
/* 129 */   BLOCK_ID_IO_Read,
/* 130 */   BLOCK_ID_IO_Write,
/* 131 */   BLOCK_ID_IO,
  
/* 132 */   BLOCK_ID_MPI_Win_create,
/* 133 */   BLOCK_ID_MPI_Win_free,
/* 134 */   BLOCK_ID_MPI_Put,
/* 135 */   BLOCK_ID_MPI_Get,
/* 136 */   BLOCK_ID_MPI_Accumulate,
/* 137 */   BLOCK_ID_MPI_Win_fence,
/* 138 */   BLOCK_ID_MPI_Win_complete,
/* 139 */   BLOCK_ID_MPI_Win_start,
/* 140 */   BLOCK_ID_MPI_Win_post,
/* 141 */   BLOCK_ID_MPI_Win_wait,
/* 142 */   BLOCK_ID_MPI_Win_test,
/* 143 */   BLOCK_ID_MPI_Win_lock,
/* 144 */   BLOCK_ID_MPI_Win_unlock,

/* 145 */   BLOCK_ID_MPI_Init_thread,

/* 146 */   BLOCK_ID_LAPI_Init,
/* 147 */   BLOCK_ID_LAPI_Term,
/* 148 */   BLOCK_ID_LAPI_Put,
/* 149 */   BLOCK_ID_LAPI_Get,
/* 150 */   BLOCK_ID_LAPI_Fence,
/* 151 */   BLOCK_ID_LAPI_Gfence,
/* 152 */   BLOCK_ID_LAPI_Address_init,
/* 153 */   BLOCK_ID_LAPI_Amsend,
/* 154 */   BLOCK_ID_LAPI_Rmw,
/* 155 */   BLOCK_ID_LAPI_Waitcntr
  
} DimBlock;
#endif

/* ==========================================================================
   ==== MPI Dimemas Collective Communications Identifiers
   ========================================================================== */

typedef enum
{
  GLOP_ID_NULL               = -1,
  GLOP_ID_MPI_Barrier        = 0,
  GLOP_ID_MPI_Bcast          = 1,
  GLOP_ID_MPI_Gather         = 2,
  GLOP_ID_MPI_Gatherv        = 3,
  GLOP_ID_MPI_Scatter        = 4,
  GLOP_ID_MPI_Scatterv       = 5,
  GLOP_ID_MPI_Allgather      = 6,
  GLOP_ID_MPI_Allgatherv     = 7,
  GLOP_ID_MPI_Alltoall       = 8,
  GLOP_ID_MPI_Alltoallv      = 9,
  GLOP_ID_MPI_Reduce         = 10,
  GLOP_ID_MPI_Allreduce      = 11,
  GLOP_ID_MPI_Reduce_scatter = 12,
  GLOP_ID_MPI_Scan           = 13

} DimCollectiveOp;

typedef enum
{
	MPI_TYPE = 1,
	MISC_TYPE,
	COMM_ALIAS_TYPE,
	OPENMP_TYPE,
	PTHREAD_TYPE,
	TRT_TYPE
} EventType_t;

EventType_t getEventType (unsigned EvType, unsigned *Type);

#endif /* __EVENTS_H_INCLUDED__ */
