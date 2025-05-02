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

#ifndef OMP_STATS_DEFINED
#define OMP_STATS_DEFINED

#include "stats_types.h"
#include "omp_utils.h"
#include "common.h"

/**
 * Enumeration of OpenMP runtime statistic IDs.
 * 
 * These IDs are used to define the types numbering used to emmit
 * the statistics events.
 * Each ID corresponds to a specific OpenMP runtime behavior category.
 */
enum {
  OMP_BURST_STATS_TIME_IN_RUNNING = 0,   // Time spent executing user code
  OMP_BURST_STATS_TIME_IN_SYNC,          // Time spent in synchronization operations
  OMP_BURST_STATS_TIME_IN_OVERHEAD,      // Time spent in OpenMP runtime overhead
  OMP_BURST_STATS_RUNNING_COUNT,         // Number of times user code was executed
  OMP_BURST_STATS_SYNC_COUNT,            // Number of synchronization operations
  OMP_BURST_STATS_OVERHEAD_COUNT,        // Number of overhead operations
  OMP_BURST_STATS_COUNT                  // Total number of statistic types
};


/**
 * Enumeration of OpenMP runtime statistic categories.
 * These categories correspond to how OpenMP calls are represented as states
 * in the new GOMP implementation.
 *
 * The statistics are divided into three main categories and stored in arrays. 
 * This enum is used for indexing this array. 
 */
enum
{
  RUNNING,          // Time spent executing actual work ( outlined parallel regions, tasks, etc.)
  SYNCHRONIZATION,  // Time spent in barriers, critical sections, and other sync operations
  OVERHEAD,         // Time spent in runtime overhead (e.g., task scheduling, region setup)
  N_OMP_CATEGORIES  // Total number of categories
};


/**
 * Structure that holds per-thread OpenMP runtime statistics.
 * 
 * Each field is an array indexed by an OpenMP region category.
 * This allows tracking statistics for different runtime events or phases.
 */
typedef struct stats_omp_thread_data
{
  UINT64 begin_time[N_OMP_CATEGORIES];   // Start time for each OpenMP region category
  UINT64 elapsed_time[N_OMP_CATEGORIES]; // Total accumulated time spent in each region
  int count[N_OMP_CATEGORIES];           // Number of times each region category was entered
} stats_omp_thread_data_t;


/**
 * OpenMP-specific statistics structure.
 * 
 * The first field must be of type 'xtr_stats_t' to maintain compatibility with the stats manager.
 * 
 * The 'common_stats_fields' field stores per-thread OpenMP runtime statistics.
 * Specifically, 'common_stats_fields.data' points to an array of 'stats_omp_thread_data_t' 
 * structures, with one entry per thread.
 * 
 * The 'open_regions_stack' field is an array of stacks (one per thread), used to track nested 
 * OpenMP region calls during execution.
 */
typedef struct xtr_OpenMP_stats
{
  xtr_stats_t common_stats_fields;   // Common statistics fields required by the stats manager
  unsigned int size;                 // Number of elements in 'common_stats_fields' and 'open_regions_stack' (used for memory allocation)
  struct stack **open_regions_stack; // Per-thread stacks for tracking nested region calls
} xtr_OpenMP_stats_t;


void *xtr_stats_OMP_init( void );
void xtr_stats_OMP_realloc ( xtr_stats_t *stats, unsigned int new_num_threads );
void xtr_stats_OMP_reset(unsigned int threadid, xtr_stats_t *stats);
xtr_stats_t *xtr_stats_OMP_dup ( xtr_stats_t *stats );
void xtr_stats_OMP_copy(unsigned int threaid, xtr_stats_t *stats_origin, xtr_stats_t *stats_destination);
void xtr_stats_OMP_subtract(unsigned int threadid, xtr_stats_t *stats, xtr_stats_t *subtrahend, xtr_stats_t *destiantion);
unsigned int xtr_stats_OMP_get_values(unsigned int threadid, xtr_stats_t *stats, INT32 *out_statistic_type, UINT64 *out_values);
unsigned int xtr_stats_OMP_get_positive_values(unsigned int threadid, xtr_stats_t *stats, INT32 *out_statistic_type, UINT64 *out_values);
stats_info_t *xtr_stats_OMP_get_types_and_descriptions( void );

void xtr_stats_OMP_update_par_OL_entry (void);
void xtr_stats_OMP_update_par_OL_exit (void);
void xtr_stats_OMP_update_synchronization_entry(void);
void xtr_stats_OMP_update_synchronization_exit(void);
void xtr_stats_OMP_update_overhead_entry(void);
void xtr_stats_OMP_update_overhead_exit(void);

void xtr_stats_OMP_free( xtr_stats_t *omp_stats );
void xtr_print_debug_omp_stats(unsigned int threadid);

#endif /* End of OMP_STATS_DEFINED */
