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

//OpenMP statistics ids
enum {
  OMP_BURST_STATS_TIME_IN_RUNNING = 0,
  OMP_BURST_STATS_TIME_IN_SYNC,
  OMP_BURST_STATS_TIME_IN_OVERHEAD,
  OMP_BURST_STATS_RUNNING_COUNT,
  OMP_BURST_STATS_SYNC_COUNT,
  OMP_BURST_STATS_OVERHEAD_COUNT,
  OMP_BURST_STATS_COUNT /* Total number of OMP statistics */
};

//These statistics are divided in three categories to store them in an array
enum
{
  RUNNING,
  SYNCHRONIZATION,
  OVERHEAD,
  N_OMP_CATEGORIES
};

/**
 * definition of the struct that contains the statistics data
 */
typedef struct stats_omp_thread_data
{
  UINT64 begin_time[N_OMP_CATEGORIES];
  UINT64 elapsed_time[N_OMP_CATEGORIES];
  int count[N_OMP_CATEGORIES];
}stats_omp_thread_data_t;

/**
 * Statistic object, first field is  'xtr_stats_t' as required by the stats manager.
 */
typedef struct xtr_OpenMP_stats
{
  xtr_stats_t common_stats_fields;
  unsigned int num_threads; //necessary to free the allocated memory
  struct stack **open_region; // used to keep track of the nested runtime calls
}xtr_OpenMP_stats_t;


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
