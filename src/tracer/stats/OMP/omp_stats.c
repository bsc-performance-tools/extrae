

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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "omp_stats.h"

#include "common.h"
#include "change_mode.h"
#include "omp_utils.h"
#include "threadid.h"
#include "xalloc.h"
#include "wrapper.h"

// #define DEBUG 

#if defined(DEBUG)
#define DEBUG_MSG(format, args...) \
	{ if(TASKID == 0 && THREADID == 0) fprintf (stderr, PACKAGE_NAME " [DEBUG] : " format " \n", ## args); }
#else
#define DEBUG_MSG(format, args...)
#endif

/**
 * Description table for OpenMP satistics
 * The address of this table will be returned when requesting 'xtr_stats_OMP_get_types_and_descriptions'
 */
static stats_info_t OMP_stats_info[] = {
  {OMP_BURST_STATS_TIME_IN_RUNNING,  "Elapsed time in useful inside parallel regions"},
  {OMP_BURST_STATS_TIME_IN_SYNC,     "Elapsed time in OpenMP synchronization"},
  {OMP_BURST_STATS_TIME_IN_OVERHEAD, "Elapsed time in OpenMP runtime overhead"},
  {OMP_BURST_STATS_RUNNING_COUNT,    "Number of useful bursts inside parallel regions"},
  {OMP_BURST_STATS_SYNC_COUNT,       "Number of OpenMP synchronization regions"},
  {OMP_BURST_STATS_OVERHEAD_COUNT,   "Number of OpenMP overhead regions"},
  {-1, NULL}
};

xtr_OpenMP_stats_t * OpenMP_stats = NULL;

xtr_OpenMP_stats_t * xtr_omp_stats_new( int iscopy );
stats_omp_thread_data_t *xtr_get_omp_stats_data( xtr_OpenMP_stats_t * omp_stats , unsigned int threadid);

/**
 * @brief Initializes OpenMP statistics.
 * 
 * This function initializes OpenMP statistics by creating a new OpenMP statistics object in which 
 *   it will store all the statistics regarding OpenMP during the execution.
 *
 * @return void* Pointer to the newly initialized OpenMP statistics object.
 *
 * @note The caller must handle the returned pointer appropriately, ensuring proper cleanup if needed.
 */
void *xtr_stats_OMP_init( void )
{
  OpenMP_stats = xtr_omp_stats_new(0);
  return OpenMP_stats;
}

/**
 * @brief Creates a new OpenMP statistics object.
 * 
 * This function allocates memory and initializes a new OpenMP statistics object.
 *
 * @param iscopy Flag indicating whether the object will be used to store copies.
 * @return xtr_OpenMP_stats_t* Pointer to the newly created OpenMP statistics object.
 *
 * @note The routine skips some memory allocations from variables that wont be used if the object will only store copies.
 * @note The caller must handle the returned pointer appropriately, ensuring proper cleanup if needed.
 */
xtr_OpenMP_stats_t * xtr_omp_stats_new( int iscopy )
{
	xtr_OpenMP_stats_t * omp_stats = xmalloc_and_zero( sizeof(xtr_OpenMP_stats_t) );
	omp_stats->num_threads = Backend_getMaximumOfThreads();

	if (omp_stats == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
		exit(-1);
	}

  omp_stats->common_stats_fields.category = OMP_STATS_GROUP;
  omp_stats->common_stats_fields.data = xmalloc_and_zero ( sizeof(stats_omp_thread_data_t ) * omp_stats->num_threads );

  if(!iscopy)
  {
    omp_stats->open_region = xmalloc_and_zero ( sizeof(struct stack *) * omp_stats->num_threads );

    for (unsigned int i = 0; i<omp_stats->num_threads; ++i)
    {
      (omp_stats->open_region)[i] = newStack();
    }
  }

	return omp_stats;
}

/**
 * @brief Reallocates resources for an OpenMP statistics object.
 * 
 * This function reallocates resources for an OpenMP statistics object to accommodate a new number of threads.
 *
 * @param omp_stats Pointer to the OpenMP statistics object to be reallocated.
 * @param new_num_threads The new number of threads to allocate resources for.
 * 
 * @note The caller must ensure that the OpenMP statistics object pointer is valid.
 */
void xtr_stats_OMP_realloc( xtr_stats_t *stats, unsigned int new_num_threads )
{
  xtr_OpenMP_stats_t *omp_stats = (xtr_OpenMP_stats_t *)stats;
  if(new_num_threads <= omp_stats->num_threads || !TRACING_OMP_STATISTICS)
    return;

  if ( omp_stats != NULL )
  {
    if(omp_stats->common_stats_fields.category == OMP_STATS_GROUP)
    {
      omp_stats->common_stats_fields.data = xrealloc(omp_stats->common_stats_fields.data, sizeof(stats_omp_thread_data_t ) * new_num_threads);
      memset(&((stats_omp_thread_data_t *)(omp_stats->common_stats_fields.data))[omp_stats->num_threads], 0, (new_num_threads-omp_stats->num_threads) * sizeof(stats_omp_thread_data_t));

      omp_stats->open_region = xrealloc(omp_stats->open_region, sizeof(struct stack *) * new_num_threads);
      for (unsigned int i = omp_stats->num_threads; i<new_num_threads; ++i)
      {
        omp_stats->open_region[i] = newStack();
      }
    }
  }
}

/**
 * @brief Resets OpenMP statistics for a specific thread.
 * 
 * This function resets OpenMP statistics for a specific thread by zeroing out all the statistics values.
 *
 * @param omp_stats Pointer to the OpenMP statistics object.
 * @param threadid The ID of the thread.
 *
 * @note The caller must ensure the validity of the OpenMP statistics object pointer and the thread ID.
 */
void xtr_stats_OMP_reset(unsigned int threadid, xtr_stats_t *stats)
{
  if(TRACING_OMP_STATISTICS)
  {
    xtr_OpenMP_stats_t *omp_stats = (xtr_OpenMP_stats_t *)stats;
    stats_omp_thread_data_t *origin_data = xtr_get_omp_stats_data(omp_stats, threadid);
    if(origin_data != NULL)
    {
      origin_data->elapsed_time[RUNNING] = 0;
      origin_data->elapsed_time[SYNCHRONIZATION] = 0;
      origin_data->elapsed_time[OVERHEAD] = 0;

      origin_data->begin_time[RUNNING] = 0;
      origin_data->begin_time[SYNCHRONIZATION] = 0;
      origin_data->begin_time[OVERHEAD] = 0;

      origin_data->count[RUNNING] = 0;
      origin_data->count[SYNCHRONIZATION] = 0;
      origin_data->count[OVERHEAD] = 0;
    }
  }
}

/**
 * @brief Duplicates OpenMP statistics object.
 * 
 * This function duplicates an OpenMP statistics object.
 *
 * @param omp_stats Pointer to the OpenMP statistics object to be duplicated.
 * @return xtr_OpenMP_stats_t* Pointer to the duplicated OpenMP statistics object, or NULL if tracing is not enabled or if the input object is NULL.
 *
 * @note The caller must handle the returned pointer appropriately, ensuring proper cleanup if needed.
 */
xtr_stats_t *xtr_stats_OMP_dup( xtr_stats_t *stats )
{
  xtr_OpenMP_stats_t *omp_stats = (xtr_OpenMP_stats_t *)stats;
  xtr_OpenMP_stats_t *new_omp_stats = NULL;
  if (TRACING_OMP_STATISTICS && omp_stats != NULL)
  {
    new_omp_stats = xtr_omp_stats_new(1);
    new_omp_stats->num_threads = omp_stats->num_threads;
    new_omp_stats->common_stats_fields.category = OMP_STATS_GROUP;
    for (unsigned int i = 0; i < new_omp_stats->num_threads; i++)
    {
      xtr_stats_OMP_copy(i, (xtr_stats_t *)omp_stats, (xtr_stats_t *)new_omp_stats);
    }
  }
  return (xtr_stats_t *)new_omp_stats;
}

/**
 * @brief Copies OpenMP statistics from one object to another for a specific thread.
 * 
 * This function copies OpenMP statistics from one object to another for a specific thread.
 *
 * @param stats_origin Pointer to the source OpenMP statistics object.
 * @param stats_destination Pointer to the destination OpenMP statistics object.
 * @param threadid The ID of the thread.
 *
 * @note The caller must ensure the validity of the OpenMP statistics object pointers and the thread ID.
 */
void xtr_stats_OMP_copy(unsigned int threadid, xtr_stats_t *stats_origin, xtr_stats_t *stats_destination)
{
  xtr_OpenMP_stats_t *omp_stats_origin = (xtr_OpenMP_stats_t *)stats_origin;
  xtr_OpenMP_stats_t *omp_stats_destination = (xtr_OpenMP_stats_t *)stats_destination;

  if( !TRACING_OMP_STATISTICS || threadid >= omp_stats_origin->num_threads || threadid >= omp_stats_destination->num_threads )
    return;

  stats_omp_thread_data_t *origin_data = xtr_get_omp_stats_data(omp_stats_origin, threadid);
  stats_omp_thread_data_t *destination_data = xtr_get_omp_stats_data(omp_stats_destination, threadid);

  if (origin_data != NULL && destination_data != NULL)
  {
    memcpy( destination_data, origin_data, sizeof(stats_omp_thread_data_t) );
  }
}


/**
 * @brief Retrieves the stack of regions for a specific thread from OpenMP statistics.
 * 
 * This function retrieves the stack of regions that keep track of the nested calls to the runtime.
 * 
 * @param omp_stats Pointer to the OpenMP statistics object.
 * @param threadid The ID of the thread.
 * @return struct stack* Pointer to the stack of regions for the specified thread, or NULL if the object or thread is not found.
 *
 * @note The caller must ensure the validity of the OpenMP statistics object pointer and the thread ID.
 */
struct stack *xtr_get_regions_stack( xtr_OpenMP_stats_t *omp_stats, unsigned int threadid )
{
  if(TRACING_OMP_STATISTICS && omp_stats != NULL)
  {
    if(omp_stats->common_stats_fields.category == OMP_STATS_GROUP)
    {
      return (omp_stats->open_region)[threadid];
    }
  }

  return NULL;
}

/**
 * @brief Retrieves OpenMP statistics for a specific thread.
 * 
 * This function retrieves the statistics data for a specific thread from 'omp_stats'.
 *
 * @param omp_stats Pointer to the OpenMP statistics object.
 * @param threadid The ID of the thread.
 * @return stats_omp_thread_data_t* Pointer to the statistics data for the specified thread, or NULL if the object or thread is not found.
 *
 * @note The caller must ensure the validity of the OpenMP statistics object pointer and the thread ID.
 */
stats_omp_thread_data_t *xtr_get_omp_stats_data( xtr_OpenMP_stats_t *omp_stats, unsigned int threadid )
{
  if(omp_stats != NULL)
  {
    if(omp_stats->common_stats_fields.category == OMP_STATS_GROUP)
    {
      return &(((stats_omp_thread_data_t *)(omp_stats->common_stats_fields.data))[threadid]);
    }
  }
  return NULL;
}

/**
 * @brief Subtracts OpenMP statistics of one object from another for a specific thread.
 *
 * @param omp_stats Pointer to the OpenMP statistics object from which to subtract statistics
 * @param subtrahend Pointer to the OpenMP statistics object to be subtracted.
 * @param destination Pointer to the destination OpenMP statistics object where the result will be stored.
 * @param threadid The ID of the thread.
 *
 * @note The caller must ensure the validity of the OpenMP statistics object pointers and the thread ID.
 */
void xtr_stats_OMP_subtract(unsigned int threadid, xtr_stats_t *stats, xtr_stats_t *subtrahend, xtr_stats_t *destination)
{
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t *omp_stats_data = xtr_get_omp_stats_data((xtr_OpenMP_stats_t *)stats, threadid);
    stats_omp_thread_data_t *subtrahend_data = xtr_get_omp_stats_data((xtr_OpenMP_stats_t *)subtrahend, threadid);
    stats_omp_thread_data_t *destination_data = xtr_get_omp_stats_data((xtr_OpenMP_stats_t *)destination, threadid);

    if (omp_stats_data != NULL && subtrahend_data != NULL && destination_data != NULL)
    {
      for (unsigned int i=0; i< N_OMP_CATEGORIES; ++i)
      {
        destination_data->elapsed_time[i] = omp_stats_data->elapsed_time[i] - subtrahend_data->elapsed_time[i];
        destination_data->count[i] = omp_stats_data->count[i] - subtrahend_data->count[i];
      }
    }
  }
}

/**
 * @brief Retrieves OpenMP statistics values and types for a specific thread.
 * 
 * @param omp_stats Pointer to the OpenMP statistics object.
 * @param out_statistic_type Pointer to the array to store statistic types.
 * @param out_values Pointer to the array to store statistic values.
 * @param threadid The ID of the thread.
 * @return unsigned int The number of events retrieved.
 * 
 * @note The caller must ensure the validity of the OpenMP statistics object pointer and the thread ID, as well as allocate sufficient memory for the output arrays.
 */
unsigned int xtr_stats_OMP_get_values(unsigned int threadid, xtr_stats_t *omp_stats, INT32 *out_statistic_type, UINT64 *out_values)
{
  unsigned int nevents = 0;
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t *thread_data = xtr_get_omp_stats_data((xtr_OpenMP_stats_t *)omp_stats, threadid);
    if (thread_data != NULL)
    {
      out_values[nevents] = thread_data->elapsed_time[RUNNING];
      out_statistic_type[nevents++] = OMP_BURST_STATS_TIME_IN_RUNNING;
      out_values[nevents] = thread_data->elapsed_time[SYNCHRONIZATION];
      out_statistic_type[nevents++] = OMP_BURST_STATS_TIME_IN_SYNC;
      out_values[nevents] = thread_data->elapsed_time[OVERHEAD];
      out_statistic_type[nevents++] = OMP_BURST_STATS_TIME_IN_OVERHEAD;
      out_values[nevents] = thread_data->count[RUNNING];
      out_statistic_type[nevents++] = OMP_BURST_STATS_RUNNING_COUNT;
      out_values[nevents] = thread_data->count[SYNCHRONIZATION];
      out_statistic_type[nevents++] = OMP_BURST_STATS_SYNC_COUNT;
      out_values[nevents] = thread_data->count[OVERHEAD];
      out_statistic_type[nevents++] = OMP_BURST_STATS_OVERHEAD_COUNT;
    }
  }

  return nevents;
}

/**
 * @brief Retrieves OpenMP statistics with positive values and their corresponding types for a specific thread.
 * 
 * @param omp_stats Pointer to the OpenMP statistics object.
 * @param out_statistic_type Pointer to the array to store statistic types.
 * @param out_values Pointer to the array to store statistic values.
 * @param threadid The ID of the thread.
 * @return int The number of positive values retrieved.
 * 
 * @note The caller must ensure the validity of the OpenMP statistics object pointer and the thread ID, as well as allocate sufficient memory for the output arrays.
 */
unsigned int xtr_stats_OMP_get_positive_values(unsigned int threadid, xtr_stats_t *omp_stats, INT32 *out_statistic_type, UINT64 *out_values)
{
  unsigned int nevents = 0;
  
  if (TRACING_OMP_STATISTICS && out_statistic_type != NULL && out_values != NULL)
  {
    stats_omp_thread_data_t *thread_data = xtr_get_omp_stats_data((xtr_OpenMP_stats_t *)omp_stats, threadid);
    if (thread_data != NULL)
    {
      if ((out_values[nevents] = thread_data->elapsed_time[RUNNING]) > 0)
      {
        out_statistic_type[nevents++] = OMP_BURST_STATS_TIME_IN_RUNNING;
      }
      if ((out_values[nevents] = thread_data->elapsed_time[SYNCHRONIZATION]) > 0)
      {
        out_statistic_type[nevents++] = OMP_BURST_STATS_TIME_IN_SYNC;
      }
      if ((out_values[nevents] = thread_data->elapsed_time[OVERHEAD]) > 0)
      {
        out_statistic_type[nevents++] = OMP_BURST_STATS_TIME_IN_OVERHEAD;
      }
      if ((out_values[nevents] = thread_data->count[RUNNING]) > 0)
      {
        out_statistic_type[nevents++] = OMP_BURST_STATS_RUNNING_COUNT;
      }
      if ((out_values[nevents] = thread_data->count[SYNCHRONIZATION]) > 0)
      {
        out_statistic_type[nevents++] = OMP_BURST_STATS_SYNC_COUNT;
      }
      if ((out_values[nevents] = thread_data->count[OVERHEAD]) > 0)
      {
        out_statistic_type[nevents++] = OMP_BURST_STATS_OVERHEAD_COUNT;
      }
    }
  }

  return nevents;
}

/**
 * @brief Retrieves the description table with all the ids and their labels.
 * 
 */
stats_info_t *xtr_stats_OMP_get_types_and_descriptions( void )
{
  return (stats_info_t *)&OMP_stats_info;
}

/**
 * UPDATE ROUTINES
 * These routines increase the statistics values of the different counters we keep track for OPENMP
 * These counters are divided into three categories: computation regions, syncronization regions
 * and runtime overhead regions.
*/

/**
 * @brief Updates OpenMP statistics upon entering a parallel or outlined region.
 * 
 * This function updates OpenMP statistics when entering a parallel or outlined region.
 * Adds the duration of the previous omp region if this call is nested.
 *
 * @note The caller must ensure that OpenMP statistics tracing is enabled and handle the initialization of required data structures.
 */
void xtr_stats_OMP_update_par_OL_entry ( void )
{
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t * thread_data = xtr_get_omp_stats_data(OpenMP_stats, THREADID);
    if(thread_data != NULL)
    {
      struct stack * open_region = xtr_get_regions_stack(OpenMP_stats, THREADID);

      if(!isEmpty(open_region))
      {
        thread_data->elapsed_time[peek(open_region)] += LAST_READ_TIME - thread_data->begin_time[peek(open_region)];
        thread_data->count[peek(open_region)] += 1;
        thread_data->begin_time[peek(open_region)] = 0;
      }

      push(open_region, RUNNING);

      thread_data->begin_time[RUNNING] = LAST_READ_TIME;
    }
  }
}

/**
 * @brief Updates OpenMP statistics upon exiting a parallel or outlined region.
 * 
 * increases the accumulated elapsed time and the number of occurrences of outlined regions 
 * 
 * @note The caller must ensure that OpenMP statistics tracing is enabled and handle the initialization of required data structures.
 */
void xtr_stats_OMP_update_par_OL_exit ( void )
{
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t * thread_data = xtr_get_omp_stats_data(OpenMP_stats, THREADID);
    if(thread_data != NULL)
    {
      struct stack * open_region = xtr_get_regions_stack(OpenMP_stats, THREADID);

      if( peek(open_region) == RUNNING && thread_data->begin_time[RUNNING] > 0 )
      {
        thread_data->elapsed_time[RUNNING] += LAST_READ_TIME - thread_data->begin_time[RUNNING];
        thread_data->count[RUNNING] += 1;
        thread_data->begin_time[RUNNING] = 0;
      }
      pop(open_region);

      if(!isEmpty(open_region))
      {
        thread_data->begin_time[peek(open_region)] = LAST_READ_TIME;
      }
    }
  }
}

/**
 * @brief Updates OpenMP statistics upon entering a synchronization region.
 * 
 * @note The caller must ensure that OpenMP statistics tracing is enabled and handle the initialization of required data structures.
 */
void xtr_stats_OMP_update_synchronization_entry( void )
{
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t * thread_data = xtr_get_omp_stats_data(OpenMP_stats, THREADID);
    if(thread_data != NULL)
    {
      struct stack * open_region = xtr_get_regions_stack(OpenMP_stats, THREADID);

      if(!isEmpty(open_region))
      {
        thread_data->elapsed_time[peek(open_region)] += LAST_READ_TIME - thread_data->begin_time[peek(open_region)];
        thread_data->count[peek(open_region)] += 1;
        thread_data->begin_time[peek(open_region)] = 0;
      }

      push(open_region, SYNCHRONIZATION);

      thread_data->begin_time[SYNCHRONIZATION] = LAST_READ_TIME;
    }
  }
}

/**
 * @brief Updates OpenMP statistics upon exiting a synchronization region.
 * 
 * @note The caller must ensure that OpenMP statistics tracing is enabled and handle the initialization of required data structures.
 */
void xtr_stats_OMP_update_synchronization_exit( void )
{
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t * thread_data = xtr_get_omp_stats_data(OpenMP_stats, THREADID);
    if(thread_data != NULL)
    {
      struct stack * open_region = xtr_get_regions_stack(OpenMP_stats, THREADID);

      if( peek(open_region) == SYNCHRONIZATION && thread_data->begin_time[SYNCHRONIZATION] > 0 )
      {
        thread_data->elapsed_time[SYNCHRONIZATION] += LAST_READ_TIME - thread_data->begin_time[SYNCHRONIZATION];
        thread_data->count[SYNCHRONIZATION] += 1;
        thread_data->begin_time[SYNCHRONIZATION] = 0;
      }
      pop(open_region);

      if(!isEmpty(open_region))
      {
        thread_data->begin_time[peek(open_region)] = LAST_READ_TIME;
      }
    }
  }
}

/**
 * @brief Updates OpenMP statistics upon entering a runtime overhead region.
 * 
 * @note The caller must ensure that OpenMP statistics tracing is enabled and handle the initialization of required data structures.
 */
void xtr_stats_OMP_update_overhead_entry( void )
{
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t * thread_data = xtr_get_omp_stats_data(OpenMP_stats, THREADID);
    if(thread_data != NULL)
    {
      struct stack * open_region = xtr_get_regions_stack(OpenMP_stats, THREADID);

      if(!isEmpty(open_region))
      {
        thread_data->elapsed_time[peek(open_region)] += LAST_READ_TIME - thread_data->begin_time[peek(open_region)];
        thread_data->count[peek(open_region)] += 1;
        thread_data->begin_time[peek(open_region)] = 0;
      }

      push(open_region, OVERHEAD);

      thread_data->begin_time[OVERHEAD] = LAST_READ_TIME;
    }
  }
}

/**
 * @brief Updates OpenMP statistics upon exiting a runtime overhead region.
 * 
 * @note The caller must ensure that OpenMP statistics tracing is enabled and handle the initialization of required data structures.
 */
void xtr_stats_OMP_update_overhead_exit( void )
{
  if (TRACING_OMP_STATISTICS)
  {
    stats_omp_thread_data_t * thread_data = xtr_get_omp_stats_data(OpenMP_stats, THREADID);
    if(thread_data != NULL)
    {
      struct stack * open_region = xtr_get_regions_stack(OpenMP_stats, THREADID);

      if( peek(open_region) == OVERHEAD && thread_data->begin_time[OVERHEAD] > 0 )
      {
        thread_data->elapsed_time[OVERHEAD] += LAST_READ_TIME - thread_data->begin_time[OVERHEAD];
        thread_data->count[OVERHEAD] += 1;
        thread_data->begin_time[OVERHEAD] = 0;
      }
      pop(open_region);

      if(!isEmpty(open_region))
      {
        thread_data->begin_time[peek(open_region)] = LAST_READ_TIME;
      }
    }
  }
}

/**
 * @brief Frees resources allocated for an OpenMP statistics object.
 * 
 * This function deallocates memory and frees resources associated with an OpenMP statistics object.
 *
 * @param omp_stats Pointer to the OpenMP statistics object to be freed.
 *
 *
 * @note The caller must ensure that the OpenMP statistics object pointer is valid and that the object is no longer needed before calling this function.
 */
void xtr_stats_OMP_free(xtr_stats_t *stats)
{
  if (stats != NULL)
  {
    xtr_OpenMP_stats_t *omp_stats = (xtr_OpenMP_stats_t *)stats;
    if(omp_stats->common_stats_fields.category == OMP_STATS_GROUP)
    {
      for (unsigned int i = 0; i < omp_stats->num_threads; ++i)
      {
        deleteStack(omp_stats->open_region[i]);
      }

      xfree(omp_stats->open_region);
      xfree(omp_stats->common_stats_fields.data);
    }
  }
}

void xtr_print_debug_omp_stats( unsigned int threadid )
{
#if !defined(DEBUG)
  UNREFERENCED_PARAMETER(threadid)
  return;
#endif

  if(OpenMP_stats == NULL)
    return;

  DEBUG_MSG("[TID:%d THD:%d] number of syncronizations: %d",
		TASKID, threadid, OpenMP_stats->common_stats_fields.data[threadid].count[SYNCHRONIZATION]);
  DEBUG_MSG("[TID:%d THD:%d] number of OVERHEAD: %d",
		TASKID, threadid, OpenMP_stats->common_stats_fields.data[threadid].count[OVERHEAD]);
  DEBUG_MSG("[TID:%d THD:%d] number of RUNNING: %d",
		TASKID, threadid, OpenMP_stats->common_stats_fields.data[threadid].count[RUNNING]);
  DEBUG_MSG("[TID:%d THD:%d] elapsed time in syncronizations: %llu",
		TASKID, threadid, OpenMP_stats->common_stats_fields.data[threadid].elapsed_time[SYNCHRONIZATION]);
  DEBUG_MSG("[TID:%d THD:%d] elapsed time in OVERHEAD: %llu",
		TASKID, threadid, OpenMP_stats->common_stats_fields.data[threadid].elapsed_time[OVERHEAD]);
  DEBUG_MSG("[TID:%d THD:%d] elapsed time in RUNNING: %llu",
		TASKID, threadid, OpenMP_stats->common_stats_fields.data[threadid].elapsed_time[RUNNING]);
}
