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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __EVENTS_H_INCLUDED__
#define __EVENTS_H_INCLUDED__

#ifdef __cplusplus
extern "C" {
#endif
unsigned IsMPI (unsigned EvType);
unsigned IsPACX (unsigned EvType);
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

/* These trace options are intended to 'catch' special architectures */
#define TRACEOPTION_UNK_ARCH        (1<<10) /* Unknown */
#define TRACEOPTION_MN_ARCH         (1<<11) /* MareNostrum */
#define TRACEOPTION_BG_ARCH         (1<<12) /* BlueGene / {PL} */

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
#define HWC_DEF_EV               40000007
#define HWC_CHANGE_EV            40000008
#define HWC_EV                   40000009
#define IOSIZE_EV                40000011
#define TRACING_EV               40000012
#define SET_TRACE_EV             40000014
#define CPU_BURST_EV             40000015
#define RUSAGE_EV                40000016
#define MPI_STATS_EV             40000017
#define TRACING_MODE_EV          40000018
#define PACX_STATS_EV            40000019
#define MEMUSAGE_EV              40000020
#define USER_SEND_EV             40000021
#define USER_RECV_EV             40000022
#define RESUME_VIRTUAL_THREAD_EV 40000023
#define SUSPEND_VIRTUAL_THREAD_EV 40000024

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

#define MEMUSAGE_BASE			 46000000
enum {
   MEMUSAGE_ARENA_EV = 0,
   MEMUSAGE_HBLKHD_EV,
   MEMUSAGE_UORDBLKS_EV,
   MEMUSAGE_FORDBLKS_EV,
   MEMUSAGE_INUSE_EV,
   MEMUSAGE_EVENTS_COUNT /* Total number of memusage events */
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

#define PACX_STATS_BASE           55000000
enum {
   PACX_STATS_P2P_COMMS_EV = 0,
   PACX_STATS_P2P_BYTES_SENT_EV,
   PACX_STATS_P2P_BYTES_RECV_EV,
   PACX_STATS_GLOBAL_COMMS_EV,
   PACX_STATS_GLOBAL_BYTES_SENT_EV,
   PACX_STATS_GLOBAL_BYTES_RECV_EV,
   PACX_STATS_TIME_IN_PACX_EV, 
   PACX_STATS_EVENTS_COUNT /* Total number of PACX statistics */
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
 *   MUST be between 50000001 - 50999999
 ******************************************************************************/
#define MPI_MIN_EV                   MPI_INIT_EV
#define MPI_MAX_EV                   50999999

#define MPI_INIT_EV                  50000001
#define MPI_BSEND_EV                 50000002
#define MPI_SSEND_EV                 50000003
#define MPI_BARRIER_EV               50000004
#define MPI_BCAST_EV                 50000005
#define MPI_SEND_EV                  50000018
#define MPI_SENDRECV_EV              50000017
#define MPI_SENDRECV_REPLACE_EV      50000081
#define MPI_RECV_EV                  50000019
#define MPI_IBSEND_EV                50000020
#define MPI_ISSEND_EV                50000021
#define MPI_ISEND_EV                 50000022
#define MPI_IRECV_EV                 50000023
#define MPI_IRCV_EV                  50000025
#define MPI_TEST_EV                  50000026
#define MPI_TEST_COUNTER_EV          50000080
#define MPI_WAIT_EV                  50000027
#define MPI_CANCEL_EV                50000030
#define MPI_RSEND_EV                 50000031
#define MPI_IRSEND_EV                50000032
#define MPI_ALLTOALL_EV              50000033
#define MPI_ALLTOALLV_EV             50000034
#define MPI_ALLREDUCE_EV             50000035
#define MPI_REDUCE_EV                50000038
#define MPI_WAITALL_EV               50000039
#define MPI_WAITANY_EV               50000068
#define MPI_WAITSOME_EV              50000069
#define MPI_IRECVED_EV               50000040
#define MPI_GATHER_EV                50000041
#define MPI_GATHERV_EV               50000042
#define MPI_SCATTER_EV               50000043
#define MPI_SCATTERV_EV              50000044
#define MPI_FINALIZE_EV              50000045
#define MPI_COMM_RANK_EV             50000046
#define MPI_COMM_SIZE_EV             50000047
#define MPI_COMM_CREATE_EV           50000048
#define MPI_COMM_DUP_EV              50000049
#define MPI_COMM_SPLIT_EV            50000050
#define MPI_RANK_CREACIO_COMM_EV     50000051      /* Used to define communicators */
#define MPI_ALLGATHER_EV             50000052
#define MPI_ALLGATHERV_EV            50000053
#define MPI_CART_CREATE_EV           50000058
#define MPI_CART_SUB_EV              50000059
#define MPI_CART_COORDS_EV           50000060
#define MPI_REDUCESCAT_EV            50000062
#define MPI_SCAN_EV                  50000063
#define MPI_PROBE_EV                 50000065
#define MPI_IPROBE_EV                50000066

#define MPI_PERSIST_REQ_EV           50000070
#define MPI_START_EV                 50000071
#define MPI_STARTALL_EV              50000072
#define MPI_REQUEST_FREE_EV          50000073
#define MPI_RECV_INIT_EV             50000074
#define MPI_SEND_INIT_EV             50000075
#define MPI_BSEND_INIT_EV            50000076
#define MPI_RSEND_INIT_EV            50000077
#define MPI_SSEND_INIT_EV            50000078

#define MPI_IPROBE_COUNTER_EV        50000200
#define MPI_TIME_OUTSIDE_IPROBES_EV  50000201

#define MPI_GLOBAL_OP_SENDSIZE       (MPI_INIT_EV+100000)
#define MPI_GLOBAL_OP_RECVSIZE       (MPI_INIT_EV+100001)
#define MPI_GLOBAL_OP_ROOT           (MPI_INIT_EV+100002)
#define MPI_GLOBAL_OP_COMM           (MPI_INIT_EV+100003)

#define MPI_FILE_OPEN_EV             50000100
#define MPI_FILE_CLOSE_EV            50000101
#define MPI_FILE_READ_EV             50000102
#define MPI_FILE_READ_ALL_EV         50000103
#define MPI_FILE_WRITE_EV            50000104
#define MPI_FILE_WRITE_ALL_EV        50000105
#define MPI_FILE_READ_AT_EV          50000106
#define MPI_FILE_READ_AT_ALL_EV      50000107
#define MPI_FILE_WRITE_AT_EV         50000108
#define MPI_FILE_WRITE_AT_ALL_EV     50000109

#define MPI_GET_EV                   50000200
#define MPI_PUT_EV                   50000201

/******************************************************************************
 *   User events to trace several MPI functions.
 *   MUST be between 50000001 - 50999999
 ******************************************************************************/
#define PACX_MIN_EV                   PACX_INIT_EV
#define PACX_MAX_EV                   51999999

#define PACX_INIT_EV                  51000001
#define PACX_BSEND_EV                 51000002
#define PACX_SSEND_EV                 51000003
#define PACX_BARRIER_EV               51000004
#define PACX_BCAST_EV                 51000005
#define PACX_SEND_EV                  51000018
#define PACX_SENDRECV_EV              51000017
#define PACX_SENDRECV_REPLACE_EV      51000081
#define PACX_RECV_EV                  51000019
#define PACX_IBSEND_EV                51000020
#define PACX_ISSEND_EV                51000021
#define PACX_ISEND_EV                 51000022
#define PACX_IRECV_EV                 51000023
#define PACX_IRCV_EV                  51000025
#define PACX_TEST_EV                  51000026
#define PACX_TEST_COUNTER_EV          51000080
#define PACX_WAIT_EV                  51000027
#define PACX_CANCEL_EV                51000030
#define PACX_RSEND_EV                 51000031
#define PACX_IRSEND_EV                51000032
#define PACX_ALLTOALL_EV              51000033
#define PACX_ALLTOALLV_EV             51000034
#define PACX_ALLREDUCE_EV             51000035
#define PACX_REDUCE_EV                51000038
#define PACX_WAITALL_EV               51000039
#define PACX_WAITANY_EV               51000068
#define PACX_WAITSOME_EV              51000069
#define PACX_IRECVED_EV               51000040
#define PACX_GATHER_EV                51000041
#define PACX_GATHERV_EV               51000042
#define PACX_SCATTER_EV               51000043
#define PACX_SCATTERV_EV              51000044
#define PACX_FINALIZE_EV              51000045
#define PACX_COMM_RANK_EV             51000046
#define PACX_COMM_SIZE_EV             51000047
#define PACX_COMM_CREATE_EV           51000048
#define PACX_COMM_DUP_EV              51000049
#define PACX_COMM_SPLIT_EV            51000050
#define PACX_RANK_CREACIO_COMM_EV     51000051      /* Used to define communicators */
#define PACX_ALLGATHER_EV             51000052
#define PACX_ALLGATHERV_EV            51000053
#define PACX_CART_CREATE_EV           51000058
#define PACX_CART_SUB_EV              51000059
#define PACX_CART_COORDS_EV           51000060
#define PACX_REDUCESCAT_EV            51000062
#define PACX_SCAN_EV                  51000063
#define PACX_PROBE_EV                 51000065
#define PACX_IPROBE_EV                51000066

#define PACX_PERSIST_REQ_EV           51000070
#define PACX_START_EV                 51000071
#define PACX_STARTALL_EV              51000072
#define PACX_REQUEST_FREE_EV          51000073
#define PACX_RECV_INIT_EV             51000074
#define PACX_SEND_INIT_EV             51000075
#define PACX_BSEND_INIT_EV            51000076
#define PACX_RSEND_INIT_EV            51000077
#define PACX_SSEND_INIT_EV            51000078

#define PACX_IPROBE_COUNTER_EV        51000200
#define PACX_TIME_OUTSIDE_IPROBES_EV  51000201

#define PACX_GLOBAL_OP_SENDSIZE       (PACX_INIT_EV+100000)
#define PACX_GLOBAL_OP_RECVSIZE       (PACX_INIT_EV+100001)
#define PACX_GLOBAL_OP_ROOT           (PACX_INIT_EV+100002)
#define PACX_GLOBAL_OP_COMM           (PACX_INIT_EV+100003)

#define PACX_FILE_OPEN_EV             51000100
#define PACX_FILE_CLOSE_EV            51000101
#define PACX_FILE_READ_EV             51000102
#define PACX_FILE_READ_ALL_EV         51000103
#define PACX_FILE_WRITE_EV            51000104
#define PACX_FILE_WRITE_ALL_EV        51000105
#define PACX_FILE_READ_AT_EV          51000106
#define PACX_FILE_READ_AT_ALL_EV      51000107
#define PACX_FILE_WRITE_AT_EV         51000108
#define PACX_FILE_WRITE_AT_ALL_EV     51000109


/******************************************************************************
 *   User events for BG PERSONALITY
 ******************************************************************************/
#define BG_PERSONALITY_TORUS_X      6000
#define BG_PERSONALITY_TORUS_Y      6001
#define BG_PERSONALITY_TORUS_Z      6002
#define BG_PERSONALITY_PROCESSOR_ID 6003

/******************************************************************************
 *   User events to trace MN topology (grodrigu)
 ******************************************************************************/
#define MN_LINEAR_HOST_EVENT         3000
#define MN_LINECARD_EVENT            3001
#define MN_HOST_EVENT                3002

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

#define PTHREAD_RWLOCK_WR_EV       61000004
#define PTHREAD_RWLOCK_RD_EV       61000005
#define PTHREAD_RWLOCK_UNLOCK_EV   61000006
#define PTHREAD_MUTEX_LOCK_EV      61000007
#define PTHREAD_MUTEX_UNLOCK_EV    61000008
#define PTHREAD_COND_SIGNAL_EV     61000009
#define PTHREAD_COND_BROADCAST_EV  61000010
#define PTHREAD_COND_WAIT_EV       61000011

#define PTHREADFUNC_EV           60000020
#define PTHREADFUNC_LINE_EV      60000120

#define TRT_SPAWN_EV             62000001
#define TRT_READ_EV              62000002
#define TRT_USRFUNC_EV           62000003
#define TRT_USRFUNC_LINE_EV      62000003

#define CUDACALL_EV              63000001
#define CUDAMEMCPY_SIZE_EV       63000002
#define CUDABASE_EV              63100000
#define CUDALAUNCH_EV            63100001
#define CUDACONFIGCALL_EV        63100002
#define CUDAMEMCPY_EV            63100003
#define CUDATHREADBARRIER_EV     63100004
#define CUDASTREAMBARRIER_EV     63100005
#define CUDASTREAMCREATE_EV      63100006
#define CUDAKERNEL_GPU_EV        63200001
#define CUDACONFIGKERNEL_GPU_EV  63200002
#define CUDAMEMCPY_GPU_EV        63200003

#define CALLER_EV                70000000
#define CALLER_LINE_EV           80000000

#define MRNET_EV                 50000
#define CLUSTER_ID_EV            90000001
#define SPECTRAL_PERIOD_EV       91000001

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

#if defined(DEAD_CODE)
/* 
 * Some Ordered Values.
 */
#define IT_MARK_VAL             2
#define WAIT_BEGIN_VAL          3
#define WAIT_END_VAL            4
#endif

/* Values */
#define EVT_BEGIN                1
#define EVT_END                  0

#define STATE_ANY                -1
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
#define STATE_MEMORY_XFER        17

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
	MPI_COMM_ALIAS_TYPE,
	MISC_TYPE,
	OPENMP_TYPE,
	PTHREAD_TYPE,
	TRT_TYPE,
	PACX_TYPE,
	PACX_COMM_ALIAS_TYPE,
	CUDA_TYPE
} EventType_t;

EventType_t getEventType (unsigned EvType, unsigned *Type);

#endif /* __EVENTS_H_INCLUDED__ */
