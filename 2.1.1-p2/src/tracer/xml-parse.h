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

#ifndef _XML_PARSE_H_INCLUDED_
#define _XML_PARSE_H_INCLUDED_

#define TRACE_TAG                       ((xmlChar*) "trace")
#define TRACE_HOME                      ((xmlChar*) "home")
#define TRACE_TYPE                      ((xmlChar*) "type")
#define TRACE_PARSER_ID                 ((xmlChar*) "xml-parser-id")
#define TRACE_TYPE_PARAVER              ((xmlChar*) "paraver")
#define TRACE_TYPE_DIMEMAS              ((xmlChar*) "dimemas")
#define TRACE_INITIAL_MODE              ((xmlChar*) "initial-mode")
#define TRACE_INITIAL_MODE_DETAIL       ((xmlChar*) "detail")
#define TRACE_INITIAL_MODE_BURSTS       ((xmlChar*) "bursts")
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
#define TRACE_GATHER_MPITS              ((xmlChar*) "gather-mpits")
#define TRACE_CIRCULAR                  ((xmlChar*) "circular")
#define TRACE_PREFIX                    ((xmlChar*) "trace-prefix")
#define TRACE_MPI                       ((xmlChar*) "mpi")
#define TRACE_PACX                      ((xmlChar*) "pacx")
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
#define TRACE_PACX_STATISTICS           ((xmlChar*) "pacx-statistics")
#define TRACE_RUSAGE                    ((xmlChar*) "resource-usage")
#define TRACE_MEMUSAGE                  ((xmlChar*) "memory-usage")
#define TRACE_CELL                      ((xmlChar*) "cell")
#define TRACE_SPU_DMATAG                ((xmlChar*) "spu-dma-channel")
#define TRACE_SPU_FILESIZE              ((xmlChar*) "spu-file-size")
#define TRACE_SPU_BUFFERSIZE            ((xmlChar*) "spu-buffer-size")
#define TRACE_LIST                      ((xmlChar*) "list")
#define TRACE_USERFUNCTION              ((xmlChar*) "user-functions")
#define TRACE_SAMPLING                  ((xmlChar*) "sampling")
#define TRACE_FREQUENCY                 ((xmlChar*) "frequency")

#define TRACE_CONTROL                   ((xmlChar*) "trace-control")
#define TRACE_CONTROL_FILE              ((xmlChar*) "file")
#define TRACE_CONTROL_GLOPS             ((xmlChar*) "global-ops")
#define TRACE_REMOTE_CONTROL            ((xmlChar*) "remote-control")
#define REMOTE_CONTROL_METHOD_MRNET     ((xmlChar*) "mrnet")
#define REMOTE_CONTROL_METHOD_SIGNAL    ((xmlChar*) "signal")
#define RC_MRNET_TARGET                 ((xmlChar*) "target")
#define RC_MRNET_ANALYSIS               ((xmlChar*) "analysis")
#define RC_MRNET_START_AFTER            ((xmlChar*) "start-after")
#define RC_SIGNAL_WHICH                 ((xmlChar*) "which")
#define RC_MRNET_SPECTRAL               ((xmlChar*) "spectral")
#define RC_MRNET_CLUSTERING             ((xmlChar*) "clustering")
#define SPECTRAL_MIN_SEEN               ((xmlChar*) "min_seen")
#define SPECTRAL_MAX_PERIODS            ((xmlChar*) "max_periods")
#define SPECTRAL_NUM_ITERS              ((xmlChar*) "num_iters")
#define CLUSTERING_MAX_TASKS            ((xmlChar*) "max_tasks")
#define CLUSTERING_MAX_POINTS           ((xmlChar*) "max_points")

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
#define TRACE_MERGE_SORTADDRESSES       ((xmlChar*) "sort-address")

void Parse_XML_File (int rank, int world_size, char *filename);

#endif
