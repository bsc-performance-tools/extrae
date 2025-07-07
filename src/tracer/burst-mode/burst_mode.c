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

#if defined(NEW_OMP_SUPPORT)
#include <omp.h>
#include "omp_events.h"
#endif

#include <stdio.h>

#include "burst_mode.h"
#include "change_mode.h"
#include "clock.h"
#include "common.h"
#include "events.h"
#include "io_probe.h"
#include "malloc_probe.h"
#include "stats_module.h"
#include "syscall_probe.h"
#include "taskid.h"
#include "threadid.h"
#include "trace_hwc.h"
#include "trace_macros.h"
#include "xalloc.h"
#include "misc_wrapper.h"

/************************************************************************************************\
 *                                      BURST MODE MODULE                                       *
 ************************************************************************************************
 * This file contains the Burst Tracing Mode API. This mode allows to reduce the trace file size
 * by tracing only significant computations (bursts) while discarding the smaller ones, as well as
 * accumulating any MPI or OpenMP activity that occurs between these significant computations as
 * statistics reported alongside them. 
 * 
 * For MPI, the burst is defined between calls to the MPI runtime. For OpenMP, there are two ways
 * to define a burst: 1) Consider each chunk of work inside a parallel (this is the default mode),
 * and 2) Consider the whole parallel as a single running burst (this requires enabling the
 * "summarization" mode, see more below). In hybrid MPI+OpenMP scenarios, a valid burst can also be
 * delimited by calls between both runtimes (e.g., from the last MPI call to the next OpenMP
 * parallel).
 * In summarization mode, if a significant burst is found within a parallel outlined
 * region, that region will be discarded, as the purpose of this feature is to provide detail when
 * the burst alone is insufficient.
 * To apply the filtering, the threshold present in the XML configuration file is
 * used.
 * The BURST mode replaces the DETAIL tracing mode by implementing the following changes: In MPI,
 * within the TRACE_MPIEVENT macro, instead of inserting a buffer event, we call xtr_burst_begin
 * and xtr_burst_end (see trace_macros_mpi.h). In OpenMP, all probes are replaced by their burst 
 * counterparts, which make the aforementioned calls (see gomp_probes_burst.c). 
 */

static int burst_initialized = FALSE;

/**
 * Region identifiers to keep track from which region were events last emmited.
  */
enum
{
  BURST_REGION,
  PARALLEL_REGION,
};

/**
 * Structure containing per-thread data required for operating in burst mode
 * to track timing and region state information
 * during execution of burst and parallel code regions.
 */
typedef struct burst_mode_st
{
  iotimer_t burst_begin_time;     // Timestamp marking the start of a burst region
  iotimer_t parallel_begin_time;  // Timestamp marking the start of a summarized parallel region
  iotimer_t last_traced_time;     // Timestamp of the last emitted event
  int in_summarization_region;    // Flag indicating whether currently in a summarization region
  int last_traced_region;         // Flag indicating if the last emitted event was from a burst or parallel region
  func_ptr_t cbk_address;         // Pointer to the outlined routine for function name and line retrieval
} burst_mode_st;

struct burst_mode_st *burst_info;
int burst_info_size = 0;

/**
 * This variable holds the statistics of the entire execution
 * And will be shared by all modules that uses statistics
 * DO NOT overwrite its fields.
 */
static  xtr_stats_t **current_stats = NULL;

/**
 * These are copies of current_stats that we will use to 
 * operate and trace
*/
static  xtr_stats_t **stats_at_burst_begin = NULL;
static  xtr_stats_t **stats_at_parallel_OL_entry = NULL;
static  xtr_stats_t **stats_at_last_traced = NULL;

static  xtr_stats_t **delta_stats = NULL;  //stores the subtraction results
INT32 base_types[NUM_STATS_GROUPS];

void trace_statistics(xtr_stats_t **begin_stats, xtr_stats_t **end_stats, xtr_stats_t **delta_stats_io, int threadid, int begin_stats_is_from_parallelOL, iotimer_t begin_time, iotimer_t end_time);

/**
 * @brief module initializer
 * 
 * Allocates memory for the variables to be used and set them to zero
 * Initializes the statistic module.
 * Using the returned handle 'current_stats' by the statistics initialization, duplicates 'current_stats'
 * to create new handles that will store values at different execution points.
 * Defines the types and labels for all the statistics in use.
 * Disables other types of tracing that would conflict with the burst mode.
 */
void xtr_burst_init ( void )
{
  if ( burst_initialized ) return;

  int current_num_threads = Backend_getMaximumOfThreads();
  burst_info = xmalloc_and_zero( current_num_threads * sizeof(burst_mode_st));
  burst_info_size = current_num_threads;

  current_stats = xtr_stats_initialize();

  stats_at_burst_begin = xtr_stats_dup(current_stats);
  stats_at_last_traced = xtr_stats_dup(current_stats);
  stats_at_parallel_OL_entry = xtr_stats_dup(current_stats);
  delta_stats = xtr_stats_dup(current_stats);

  int num_stats = 0;
  stats_info_t **current_stats_info = xtr_stats_get_description_table();

  for(int i = 0; i < NUM_STATS_GROUPS; i++)
  {
    if(current_stats[i] != NULL)
    {
      base_types[i] = xtr_stats_get_category(current_stats[i]) == MPI_STATS_GROUP? MPI_BURST_STATS_BASE : OMP_BURST_STATS_BASE;
      
      for(int j = 0; current_stats_info[i][j].id != -1; j++)
        Extrae_define_event_type_Wrapper (current_stats_info[i][j].id+base_types[i], current_stats_info[i][j].description, num_stats, NULL, NULL);
    }
  }

  Extrae_set_trace_io(FALSE);
  Extrae_set_trace_malloc(FALSE);
  Extrae_set_pthread_tracing(FALSE);
  Extrae_set_trace_syscall(FALSE);
  burst_initialized = TRUE;
}

/**
 * @brief Reallocates resources for burst mode to handle an increased number of threads.
 * 
 * This function reallocates all the variables of this module and the statistics arrays to accommodate 
 * an increase in the number of threads. It ensures that the additional elements are initialized to zero
 *
 * This operation is not thread safe
 * 
 * @param new_num_threads The new number of threads.
 */
void xtr_burst_realloc(int new_num_threads)
{
  if ( burst_initialized && new_num_threads > burst_info_size )
  {
    burst_info = xrealloc(burst_info, new_num_threads * sizeof(burst_mode_st));
    memset(&burst_info[burst_info_size], 0, (new_num_threads-burst_info_size) * sizeof(burst_mode_st));
    burst_info_size = new_num_threads;

    xtr_stats_realloc( stats_at_burst_begin, new_num_threads );
    xtr_stats_realloc( stats_at_last_traced, new_num_threads );
    xtr_stats_realloc( stats_at_parallel_OL_entry, new_num_threads );
    xtr_stats_realloc( delta_stats, new_num_threads );
  }
}

/**
 * @brief starts a burst region for the current thread
 * 
 * This function initiates a new burst for the current thread. 
 * Checks if burst mode is activated.
 * Sets an initial time for 'last_traced_time'.
 * Records the opening time of the current burst. (We are exiting a runtime call)
 * 
 * Copies the current statistics to a designated statistics array.
 * Accumulates HW counter values.
 * (The statistics and HWC will be reported at the burst exit if the burst is not discarded)
 */
void xtr_burst_begin( void )
{
  if( CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURST || !burst_initialized )
    return;

  if(burst_info[THREADID].last_traced_time == 0) burst_info[THREADID].last_traced_time = LAST_READ_TIME;

  burst_info[THREADID].burst_begin_time = LAST_READ_TIME;

  xtr_stats_copyto(current_stats, stats_at_burst_begin);
  HARDWARE_COUNTERS_ACCUMULATE(THREADID, burst_info[THREADID].burst_begin_time);
}


/**
 * @brief Ends the current burst region
 *
 * Retrieves the burst begin and end times.
 * Checks if burst mode is activated, initialized and if the duration of the current burst
 * is greater than the configured threshold.
 * Traces the accumulated statistics until the begining of the current burst region.
 * Traces the 'burst begin event' along with the accumulated HW counters until the begining of the current burst region.
 * Changes the HW set if configured.
 * Traces the 'burst end event'.
 * Resets the acumulated HW counter.
 * Updates burst information for the thread, including the last traced region and time.
 * Copies current statistics to the `stats_at_last_traced` array.
 * 
 * @details
 * 
 * The calculation of the statistics of the region is done by subtracting the statistics
 * read at the last time they were reported from the ones read at the burst entry.
 * 
 * These statistics are emmited at the burst region begin time alongside with their matching 
 * 0s at the time of the last burst exit point traced.
 * 
 * The order of operations guarantees that resulting events are 
 * cronologically ordered in the resulting trace.
 * 
 * It sets 'in_summarization_region' to FALSE to forbbid the next closing parallel_OL 
 * to report statistics.
 * 
 * @return int Returns `TRUE` if the burst has been traced, `FALSE` otherwise.
 */
int xtr_burst_end( void )
{
  iotimer_t begin_time = burst_info[THREADID].burst_begin_time;
  iotimer_t end_time = LAST_READ_TIME;

  if( CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURST || !burst_initialized || begin_time == 0 || (end_time - begin_time) <= MINIMUM_BURST_DURATION )
    return FALSE;

  trace_statistics(stats_at_last_traced, stats_at_burst_begin, delta_stats, THREADID, burst_info[THREADID].last_traced_region == PARALLEL_REGION, burst_info[THREADID].last_traced_time, begin_time);

  TRACE_EVENTAND_ACCUMULATEDCOUNTERS(begin_time, CPU_BURST_EV, EVT_BEGIN);
  HARDWARE_COUNTERS_CHANGE(end_time, THREADID);
  TRACE_EVENTANDCOUNTERS(end_time, CPU_BURST_EV, EVT_END, TRUE);
  ACCUMULATED_COUNTERS_RESET(THREADID);

  burst_info[THREADID].last_traced_region = BURST_REGION;
  burst_info[THREADID].last_traced_time = end_time;
  burst_info[THREADID].in_summarization_region = FALSE;

  xtr_stats_copyto(current_stats, stats_at_last_traced);

  return TRUE;
}

/**
 * @brief Traces the difference of the statistics between the beginning and end of a burst region.
 * 
 * This function calculates and records the statistics between the start and 
 * end of a burst region for a given thread.
 *
 * @param begin_stats Pointer to the array of statistics at the beginning of the burst.
 * @param end_stats Pointer to the array of statistics at the end of the burst.
 * @param delta_stats_io Input Output paramenter. 
 *                       As input it is a pointer to the array of statistics that stores the last emited statistics.
 *                       As output it will store the subtraction result for future consulting.
 * @param threadid The ID of the thread for which statistics are being traced.
 * @param begin_stats_is_from_parallelOL Flag indicating if the statistics from the beginning of the burst have been taken from the beggining of a parallel outlined region(summarization mode).
 * @param begin_time The timestamp marking the beginning of the burst region.
 * @param end_time The timestamp marking the end of the burst region.
 *
 * @details
 * The retrieval of the last traced types is necessary in a burst region that follows a parallel OL region because when the 
 * summarization mode is activated statistics are reported at the beginning and end of the parallel region, 
 * and we don't want to have two events, values and zeros, at the same timestamp. Therefore we retrieve 
 * the types previously emmited and avoid to emmit zeros for them.
 */
void trace_statistics(xtr_stats_t **begin_stats, xtr_stats_t **end_stats, xtr_stats_t **delta_stats_io, int threadid, int begin_stats_is_from_parallelOL, iotimer_t begin_time, iotimer_t end_time)
{
  int i = 0, j = 0, k = 0;

 // Declares arrays that will hold the types and values to be traced.
  int num_last_traced_stats[NUM_STATS_GROUPS] = {0};
  INT32 last_traced_types[NUM_STATS_GROUPS][STATS_SIZE_PER_GROUP] = {0};
  UINT64 last_traced_values[NUM_STATS_GROUPS][STATS_SIZE_PER_GROUP] = {0};

  int num_current_stats[NUM_STATS_GROUPS] = {0};
  INT32 current_types[NUM_STATS_GROUPS][STATS_SIZE_PER_GROUP] = {0};
  UINT64 current_values[NUM_STATS_GROUPS][STATS_SIZE_PER_GROUP] = {0};

  int num_zero_stats[NUM_STATS_GROUPS] = {0};
  INT32 zero_stats_types[NUM_STATS_GROUPS][STATS_SIZE_PER_GROUP] = {0};
  UINT64 zero_stats_values[NUM_STATS_GROUPS][STATS_SIZE_PER_GROUP] = {0};

  if(begin_time == end_time)
    return;

 // If 'begin_stats' has been taken in the OL parallel(summarization mode),
 // retrieves its statistics types and values and stores them in 'last_traced_xx'.
  if(begin_stats_is_from_parallelOL)
  {
    xtr_stats_get_values(threadid, delta_stats_io, num_last_traced_stats, last_traced_types, last_traced_values);
  }

 // Subtracts the begin and end statistics to obtain the ones within the region.
 // Selects types that were not emited at 'begin_time' and fills 'zero_stats_types' with them,
 // then traces these types with value 0.
  xtr_stats_subtract(threadid, end_stats, begin_stats, delta_stats_io );
  xtr_stats_get_values(threadid, delta_stats_io,  num_current_stats, current_types, current_values);

  // emits the matching types and 0s of the current statistics
  for(i=0; i <NUM_STATS_GROUPS; ++i)
  {
    for (j = 0; j < num_current_stats[i]; ++j)
    {
      for (k = 0; k < num_last_traced_stats[i]; ++k)
      {
        if(current_types[i][j] == last_traced_types[i][k])
          break;
      }

      if(k == num_last_traced_stats[i])
        zero_stats_types[i][num_zero_stats[i]++] = current_types[i][j] + base_types[i];

      //convert to paraver type
      current_types[i][j] += base_types[i];
    }

    if(num_zero_stats[i] > 0)
    {
      THREAD_TRACE_N_EVENTS(threadid, begin_time, num_zero_stats[i], zero_stats_types[i], zero_stats_values[i]);
    }
  }

  // Traces the current calculated statistics emmiting their types and their values.
  //separated loops to preserve order time in buffer insertions
  for(i = 0; i < NUM_STATS_GROUPS; ++i)
  {
    if(num_current_stats[i] > 0)
    {
      THREAD_TRACE_N_EVENTS(threadid, end_time, num_current_stats[i] , current_types[i], current_values[i]);
    }
  }
}

#if defined(NEW_OMP_SUPPORT)
/**
 * @brief Records the entry point into a burst parallel region.
 * 
 * This function logs the entry into a parallel region for the current thread. 
 * Performing the following operations:
 * - Records the parallel region begin time.
 * - Sets the summarization region status to true.
 * - Stores the callback function address.
 * - Copies the current statistics to the `stats_at_parallel_OL_entry` array.
 *
 * @param function_address The address of the function at the entry point.
 */
void xtr_burst_parallel_OL_entry( void * function_address )
{
  if( CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURST || !burst_initialized || !OMP_SUMMARIZATION_ENABLED || omp_get_level() > 1 )
    return;

  burst_info[THREADID].parallel_begin_time = LAST_READ_TIME;
  burst_info[THREADID].in_summarization_region = TRUE;
  burst_info[THREADID].cbk_address = function_address;

  xtr_stats_copyto(current_stats, stats_at_parallel_OL_entry);
}

/**
 * @brief Records the exit point from a parallel burst region
 *
 * The function performs the following operations:
 * - Retrieves the entry and exit times for the parallel region.
 * - Verifies that the duration of the parallel region exceeds the minimum burst duration.
 * - Traces statistics for the period from the last traced time to the parallel entry time.
 * - Traces the function address of the current parallel region.
 * - Traces statistics for the period from the parallel entry time to the parallel exit time.
 * - Updates the thread's last traced region and time.
 * - Marks the thread as no longer being in a summarization region.
 * - Copies current statistics to the `stats_at_last_traced` array.
 *
 */
void xtr_burst_parallel_OL_exit()
{
  iotimer_t parallel_OL_entry_time = burst_info[THREADID].parallel_begin_time;
  iotimer_t parallel_OL_exit_time = LAST_READ_TIME;

  if( CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURST || !burst_initialized || !OMP_SUMMARIZATION_ENABLED ||
      burst_info[THREADID].in_summarization_region == FALSE || omp_get_level() > 1 ||
      (parallel_OL_exit_time - parallel_OL_entry_time) <= MINIMUM_BURST_DURATION )
    return;

  trace_statistics(stats_at_last_traced, stats_at_parallel_OL_entry, delta_stats, THREADID, burst_info[THREADID].last_traced_region == PARALLEL_REGION, burst_info[THREADID].last_traced_time, parallel_OL_entry_time);
  Extrae_OpenMP_Outlined_Entry_At(parallel_OL_entry_time, burst_info[THREADID].cbk_address);

  trace_statistics(stats_at_parallel_OL_entry, current_stats, delta_stats, THREADID, TRUE, parallel_OL_entry_time, parallel_OL_exit_time);
  Extrae_OpenMP_Outlined_Exit_At(parallel_OL_exit_time);

  burst_info[THREADID].last_traced_region = PARALLEL_REGION;
  burst_info[THREADID].last_traced_time = parallel_OL_exit_time;
  burst_info[THREADID].in_summarization_region = FALSE;

  xtr_stats_copyto(current_stats, stats_at_last_traced);
}
#endif

/**
 * @brief Finalizes the burst mode tracing for all threads.
 * 
 * This function finalizes the burst mode tracing by performing cleanup operations
 * and ensuring that all remaining burst mode statistics are properly traced and recorded.
 * It deinitializes the burst mode and frees allocated resources.
 */
void xtr_burst_finalize (void)
{
  if( !burst_initialized )
    return;

  burst_initialized = FALSE;

  for (int threadid = 0; threadid < Backend_getMaximumOfThreads(); ++threadid)
  {
    if ( burst_info[threadid].last_traced_time < burst_info[threadid].burst_begin_time )
    {
      trace_statistics(stats_at_last_traced, stats_at_burst_begin, delta_stats, threadid, burst_info[threadid].last_traced_region == PARALLEL_REGION, burst_info[threadid].last_traced_time, burst_info[threadid].burst_begin_time);

      TRACE_EVENTAND_ACCUMULATEDCOUNTERS(burst_info[threadid].burst_begin_time, CPU_BURST_EV, EVT_BEGIN);
      ACCUMULATED_COUNTERS_RESET(threadid);
    }
  }

  xfree(burst_info);
  burst_info_size = 0;

  xtr_stats_free(stats_at_burst_begin);
  xtr_stats_free(stats_at_parallel_OL_entry);
  xtr_stats_free(stats_at_last_traced);
  xtr_stats_free(delta_stats);
  xtr_stats_finalize();
}

/**
 * @brief Emits accumulated statistics for all active threads.
 * 
 * This function forces the emission of all currently accumulated burst-related 
 * statistics, including runtime metrics and hardware counters, for each thread.
 */
void xtr_burst_emit_statistics (void)
{
  if( !burst_initialized )
    return;

  for (int threadid = 0; threadid < Backend_getMaximumOfThreads(); ++threadid)
  {
    if ( burst_info[threadid].last_traced_time < burst_info[threadid].burst_begin_time )
    {
      trace_statistics(stats_at_last_traced, stats_at_burst_begin, delta_stats, threadid, burst_info[threadid].last_traced_region == PARALLEL_REGION, burst_info[threadid].last_traced_time, burst_info[threadid].burst_begin_time);

      TRACE_EVENTAND_ACCUMULATEDCOUNTERS(burst_info[threadid].burst_begin_time, CPU_BURST_EV, EVT_BEGIN);
      ACCUMULATED_COUNTERS_RESET(threadid);
    }
  }
}

