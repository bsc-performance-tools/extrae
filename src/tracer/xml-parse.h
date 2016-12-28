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

#ifndef _XML_PARSE_H_INCLUDED_
#define _XML_PARSE_H_INCLUDED_

#define xmlYES               (xmlChar*) "yes"
#define xmlNO                (xmlChar*) "no"
#define xmlCOMMENT           (xmlChar*) "COMMENT"
#define xmlTEXT              (xmlChar*) "text"
#define XML_ENVVAR_CHARACTER (xmlChar)  '$'

/* Free memory if not null */
#define XML_FREE(ptr) \
        if (ptr != NULL) xmlFree(ptr);

/* master fprintf :) */
#define mfprintf \
        if (rank == 0) fprintf 

#define TRACE_TAG                       ((xmlChar*) "trace")
#define TRACE_HOME                      ((xmlChar*) "home")
#define TRACE_TYPE                      ((xmlChar*) "type")
#define TRACE_TYPE_PARAVER              ((xmlChar*) "paraver")
#define TRACE_TYPE_DIMEMAS              ((xmlChar*) "dimemas")
#define TRACE_INITIAL_MODE              ((xmlChar*) "initial-mode")
#define TRACE_INITIAL_MODE_DETAIL       ((xmlChar*) "detail")
#define TRACE_INITIAL_MODE_BURSTS       ((xmlChar*) "bursts")
#define TRACE_INITIAL_MODE_BURST        ((xmlChar*) "burst")
#define TRACE_COUNTERS                  ((xmlChar*) "counters")
#define TRACE_CALLERS                   ((xmlChar*) "callers")
#define TRACE_CPU                       ((xmlChar*) "cpu")
#define TRACE_STARTSET                  ((xmlChar*) "starting-set-distribution")
#define TRACE_HWCSET                    ((xmlChar*) "set")
#define TRACE_HWCSET_CHANGEAT_GLOBALOPS ((xmlChar*) "changeat-globalops")
#define TRACE_HWCSET_CHANGEAT_TIME      ((xmlChar*) "changeat-time")
#define TRACE_HWCSET_DOMAIN             ((xmlChar*) "domain")
#define TRACE_HWCSET_OVERFLOW_COUNTER   ((xmlChar*) "overflow-counter")
#define TRACE_HWCSET_OVERFLOW_VALUE     ((xmlChar*) "overflow-value")
#define TRACE_ENABLED                   ((xmlChar*) "enabled")
#define TRACE_SIZE                      ((xmlChar*) "size")
#define TRACE_MPI_CALLERS               ((xmlChar*) "callers")
#define TRACE_FINAL_DIR                 ((xmlChar*) "final-directory")
#define TRACE_DIR                       ((xmlChar*) "temporal-directory")
#define TRACE_MKDIR                     ((xmlChar*) "make-dir")
#define TRACE_MINIMUM_TIME              ((xmlChar*) "minimum-time")
#define TRACE_FREQUENCY                 ((xmlChar*) "frequency")
#define TRACE_PERIOD                    ((xmlChar*) "period")
#define TRACE_VARIABILITY               ((xmlChar*) "variability")
#define TRACE_TYPE                      ((xmlChar*) "type")
#define TRACE_CIRCULAR                  ((xmlChar*) "circular")
#define TRACE_PREFIX                    ((xmlChar*) "trace-prefix")
#define TRACE_MPI                       ((xmlChar*) "mpi")
#define TRACE_SHMEM                     ((xmlChar*) "shmem")
#define TRACE_OPENCL                    ((xmlChar*) "opencl")
#define TRACE_CUDA                      ((xmlChar*) "cuda")
#define TRACE_PTHREAD_LOCKS             ((xmlChar*) "locks")
#define TRACE_PTHREAD                   ((xmlChar*) "pthread")
#define TRACE_OMP_LOCKS                 ((xmlChar*) "locks")
#define TRACE_OMP                       ((xmlChar*) "openmp")
#define TRACE_STORAGE                   ((xmlChar*) "storage")
#define TRACE_BUFFER                    ((xmlChar*) "buffer")
#define TRACE_OTHERS                    ((xmlChar*) "others")
#define TRACE_BURSTS                    ((xmlChar*) "bursts")
#define TRACE_THRESHOLD                 ((xmlChar*) "threshold")
#define TRACE_NETWORK                   ((xmlChar*) "network")
#define TRACE_MPI_STATISTICS            ((xmlChar*) "mpi-statistics")
#define TRACE_RUSAGE                    ((xmlChar*) "resource-usage")
#define TRACE_MEMUSAGE                  ((xmlChar*) "memory-usage")
#define TRACE_DYNAMIC_MEMORY            ((xmlChar*) "dynamic-memory")
#define TRACE_DYNAMIC_MEMORY_ALLOC      ((xmlChar*) "alloc")
#define TRACE_DYNAMIC_MEMORY_ALLOC_THRESHOLD   ((xmlChar*) "threshold")
#define TRACE_DYNAMIC_MEMORY_FREE       ((xmlChar*) "free")
#define TRACE_IO                        ((xmlChar*) "input-output")
#define TRACE_SYSCALL                   ((xmlChar*) "syscall")
#define TRACE_LIST                      ((xmlChar*) "list")
#define TRACE_EXCLUDE_AUTOMATIC_FUNCTIONS ((xmlChar*) "exclude-automatic-functions")
#define TRACE_USERFUNCTION              ((xmlChar*) "user-functions")
#define TRACE_SAMPLING                  ((xmlChar*) "sampling")
#define TRACE_FINALIZE_ON_SIGNAL        ((xmlChar*) "finalize-on-signal")
#define TRACE_FINALIZE_ON_SIGNAL_USR1   ((xmlChar*) "SIGUSR1")
#define TRACE_FINALIZE_ON_SIGNAL_USR2   ((xmlChar*) "SIGUSR2")
#define TRACE_FINALIZE_ON_SIGNAL_INT    ((xmlChar*) "SIGINT")
#define TRACE_FINALIZE_ON_SIGNAL_QUIT   ((xmlChar*) "SIGQUIT")
#define TRACE_FINALIZE_ON_SIGNAL_TERM   ((xmlChar*) "SIGTERM")
#define TRACE_FINALIZE_ON_SIGNAL_XCPU   ((xmlChar*) "SIGXCPU")
#define TRACE_FINALIZE_ON_SIGNAL_FPE    ((xmlChar*) "SIGFPE")
#define TRACE_FINALIZE_ON_SIGNAL_SEGV   ((xmlChar*) "SIGSEGV")
#define TRACE_FINALIZE_ON_SIGNAL_ABRT   ((xmlChar*) "SIGABRT")
#define TRACE_FLUSH_SAMPLE_BUFFER_AT_INST_POINT ((xmlChar*) "flush-sampling-buffer-at-instrumentation-point")

#define TRACE_CONTROL                   ((xmlChar*) "trace-control")
#define TRACE_CONTROL_FILE              ((xmlChar*) "file")
#define TRACE_CONTROL_GLOPS             ((xmlChar*) "global-ops")
#define TRACE_REMOTE_CONTROL            ((xmlChar*) "remote-control")
#define REMOTE_CONTROL_METHOD_MRNET     ((xmlChar*) "mrnet")
#define RC_MRNET_TARGET                 ((xmlChar*) "target")
#define RC_MRNET_ANALYSIS               ((xmlChar*) "analysis")
#define RC_MRNET_START_AFTER            ((xmlChar*) "start-after")
#define REMOTE_CONTROL_METHOD_ONLINE    ((xmlChar*) "online")
#define RC_ONLINE_TYPE                  ((xmlChar*) "analysis")
#define RC_ONLINE_FREQ                  ((xmlChar*) "frequency")
#define RC_ONLINE_TOPO                  ((xmlChar*) "topology")
#define RC_ONLINE_SPECTRAL              ((xmlChar*) "spectral")
#define RC_ONLINE_SPECTRAL_ADVANCED                   ((xmlChar*) "spectral_advanced")
#define RC_ONLINE_SPECTRAL_ADVANCED_PERIODIC_ZONE     ((xmlChar*) "periodic_zone")
#define RC_ONLINE_SPECTRAL_ADVANCED_NON_PERIODIC_ZONE ((xmlChar*) "non_periodic_zone")
#define SPECTRAL_MAX_PERIODS            ((xmlChar*) "max_periods")
#define SPECTRAL_MIN_SEEN               ((xmlChar*) "min_seen")
#define SPECTRAL_NUM_ITERS              ((xmlChar*) "num_iters")
#define SPECTRAL_MIN_LIKENESS           ((xmlChar*) "min_likeness")
#define SPECTRAL_DETAIL_LEVEL           ((xmlChar*) "detail_level")
#define SPECTRAL_MIN_DURATION           ((xmlChar*) "min_duration")
#define SPECTRAL_BURST_THRESHOLD        ((xmlChar*) "burst_threshold")
#define RC_ONLINE_CLUSTERING            ((xmlChar*) "clustering")
#define CLUSTERING_MAX_TASKS            ((xmlChar*) "max_tasks")
#define CLUSTERING_MAX_POINTS           ((xmlChar*) "max_points")
#define CLUSTERING_CONFIG               ((xmlChar*) "config")
#define RC_ONLINE_GREMLINS              ((xmlChar*) "gremlins")
#define GREMLINS_START                  ((xmlChar*) "start")
#define GREMLINS_INCREMENT              ((xmlChar*) "increment")
#define GREMLINS_ROUNDTRIP              ((xmlChar*) "roundtrip")
#define GREMLINS_LOOP                   ((xmlChar*) "loop")

#define TRACE_MERGE                     ((xmlChar*) "merge")
#define TRACE_MERGE_SYNCHRONIZATION     ((xmlChar*) "synchronization")
#define TRACE_MERGE_BINARY              ((xmlChar*) "binary")
#define TRACE_MERGE_TREE_FAN_OUT        ((xmlChar*) "tree-fan-out")
#define TRACE_MERGE_MAX_MEMORY          ((xmlChar*) "max-memory")
#define TRACE_MERGE_JOINT_STATES        ((xmlChar*) "joint-states")
#define TRACE_MERGE_KEEP_MPITS          ((xmlChar*) "keep-mpits")
#define TRACE_MERGE_SYN_NODE            ((xmlChar*) "node")
#define TRACE_MERGE_SYN_TASK            ((xmlChar*) "task")
#define TRACE_MERGE_SYN_DEFAULT         ((xmlChar*) "default")
#define TRACE_MERGE_SORTADDRESSES       ((xmlChar*) "sort-addresses")
#define TRACE_MERGE_OVERWRITE           ((xmlChar*) "overwrite")

#define TRACE_PEBS_SAMPLING             ((xmlChar*) "pebs-sampling")
#define TRACE_PEBS_SAMPLING_LOADS       ((xmlChar*) "loads")
#define TRACE_PEBS_SAMPLING_STORES      ((xmlChar*) "stores")
#define TRACE_PEBS_MIN_MEM_LATENCY      ((xmlChar*) "minimum-latency")

#define TRACE_CPU_EVENTS		((xmlChar*) "cpu-events")
#define TRACE_CPU_EVENTS_FREQUENCY	((xmlChar*) "frequency")
#define TRACE_CPU_EVENTS_EMIT_ALWAYS    ((xmlChar*) "emit-always")

void Parse_XML_File (int rank, int world_size, const char *filename);

#endif
