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

#include <mpi.h>
#include <stdlib.h>
#include <string.h>

#include "mpi_stats.h"
#include "mode.h"
#include "taskid.h"
#include "threadid.h"
#include "stdio.h"
#include "common.h"
#include "xalloc.h"


// # define DEBUG

#if defined(DEBUG)
#define DEBUG_MSG(format, args...) \
	{ if(TASKID == 0) fprintf (stderr, PACKAGE_NAME " [DEBUG] : " format " \n", ## args); }
#else
#define DEBUG_MSG(format, args...)
#endif




static stats_info_t MPI_stats_info[] = {
  {MPI_BURST_STATS_P2P_COUNT, "Number of P2P MPI calls"},
  {MPI_BURST_STATS_P2P_BYTES_SENT, "Bytes sent in P2P MPI calls"},
  {MPI_BURST_STATS_P2P_BYTES_RECV, "Bytes received in P2P MPI calls"},
  {MPI_BURST_STATS_GLOBAL_COUNT, "Number of GLOBAL MPI calls"},
  {MPI_BURST_STATS_GLOBAL_BYTES_SENT, "Bytes sent in GLOBAL MPI calls"},
  {MPI_BURST_STATS_GLOBAL_BYTES_RECV, "Bytes received in GLOBAL MPI calls"},
  {MPI_BURST_STATS_TIME_IN_MPI, "Elapsed time in MPI"},
  {MPI_BURST_STATS_P2P_INCOMING_COUNT, "Number of incoming P2P MPI calls"},
  {MPI_BURST_STATS_P2P_OUTGOING_COUNT, "Number of outgoing P2P MPI calls"},
  {MPI_BURST_STATS_P2P_INCOMING_PARTNERS_COUNT, "Number of partners in incoming communications"},
  {MPI_BURST_STATS_P2P_OUTGOING_PARTNERS_COUNT, "Number of partners in outgoing communications"},
  {MPI_BURST_STATS_TIME_IN_OTHER,  "Elapsed time in OTHER MPI calls"},
  {MPI_BURST_STATS_TIME_IN_P2P, "Elapsed time in P2P MPI calls"},
  {MPI_BURST_STATS_TIME_IN_GLOBAL, "Elapsed time in GLOBAL MPI calls"},
  {MPI_BURST_STATS_OTHER_COUNT, "Number of OTHER MPI calls"},
	{-1, NULL}
};

xtr_MPI_stats_t * MPI_stats = NULL;

xtr_MPI_stats_t * xtr_mpi_stats_new();
stats_mpi_thread_data_t *xtr_get_mpi_stats_data( xtr_MPI_stats_t * mpi_stats, int threadid );
int mpi_stats_get_num_partners(stats_mpi_thread_data_t * data, int * partners_vector);

/**
 * @brief Initializes the MPI statistics by allocating and setting up a new xtr_MPI_stats_t structure.
 * 
 * This function allocates memory for a new `xtr_MPI_stats_t` structure and initializes its fields
 * by calling the `xtr_mpi_stats_new` function. The newly created structure is assigned to the global
 * `MPI_stats` variable and returned.
 * 
 * @return void* A pointer to the newly allocated and initialized `xtr_MPI_stats_t` structure.
 */
void * xtr_stats_MPI_init( void )
{
	MPI_stats = xtr_mpi_stats_new();
  return MPI_stats;
}

/**
 * @brief Allocates and initializes a new xtr_MPI_stats_t structure.
 * 
 * This function allocates memory for a new `xtr_MPI_stats_t` structure and initializes its fields,
 * including allocating memory for `stats_mpi_thread_data_t` structures and their associated P2P
 * partner arrays. Sets all new memory to zero.
 * 
 * @return xtr_MPI_stats_t* A pointer to the newly allocated and initialized `xtr_MPI_stats_t` structure.
 */
xtr_MPI_stats_t * xtr_mpi_stats_new()
{
	int num_tasks = Extrae_get_num_tasks();

	xtr_MPI_stats_t * mpi_stats = xmalloc( sizeof(xtr_MPI_stats_t) );
	mpi_stats->num_threads = Backend_getMaximumOfThreads();

	if (mpi_stats == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
		exit(-1);
	}

	mpi_stats->common_stats_field.category = MPI_STATS_GROUP;
	mpi_stats->common_stats_field.data = xmalloc_and_zero (sizeof(stats_mpi_thread_data_t) * mpi_stats->num_threads);

	for(int i =0; i < mpi_stats->num_threads ; ++i)
	{
		((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_In = xmalloc_and_zero (num_tasks * sizeof(int));
		if (((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_In == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
			exit(-1);
		}
		((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_Out = xmalloc_and_zero (num_tasks * sizeof(int));
		if (((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_Out == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
			exit(-1);
		}
	}

	return mpi_stats;
}

/**
 * @brief Reallocates the memory for MPI statistics to accommodate a new number of threads.
 * 
 * This function increases the capacity of the MPI statistics structure to handle a greater number
 * of threads. It reallocates memory for the `common_stats_field.data` field and initializes the
 * `P2P_Partner_In` and `P2P_Partner_Out` arrays for the newly added threads.
 * 
 * @param mpi_stats Pointer to the `xtr_MPI_stats_t` structure to be reallocated.
 * @param new_num_threads The new number of threads to accommodate.
 */
void xtr_stats_MPI_realloc (xtr_MPI_stats_t *mpi_stats, int new_num_threads )
{
	if(new_num_threads <= mpi_stats->num_threads)
	 return;

	int num_tasks = Extrae_get_num_tasks();
	if(mpi_stats != NULL)
	{
		if(mpi_stats->common_stats_field.category == MPI_STATS_GROUP)
		{
			mpi_stats->common_stats_field.data = xrealloc(mpi_stats->common_stats_field.data, sizeof(stats_mpi_thread_data_t) * new_num_threads);
      memset(&mpi_stats->common_stats_field.data[mpi_stats->num_threads], 0, (new_num_threads-mpi_stats->num_threads) * sizeof(stats_mpi_thread_data_t));

			for(int i=mpi_stats->num_threads; i < new_num_threads ; ++i)
			{
				((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_In = xmalloc_and_zero (num_tasks * sizeof(int));
				if (((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_In == NULL)
				{
					fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
					exit(-1);
				}
				((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_Out = xmalloc_and_zero (num_tasks * sizeof(int));
				if (((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_Out == NULL)
				{
					fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for MPI Stats");
					exit(-1);
				}
			}
			mpi_stats->num_threads = new_num_threads;
		}
	}
}

/**
 * @brief Resets the MPI statistics data for a specific thread.
 *
 * This function resets all the MPI statistics fields for the specified thread to 0.
 * If `TRACING_MPI_STATISTICS` is enabled and the `origin_data` is not NULL, it clears all counters and timings.
 *
 * @param mpi_stats Pointer to the `xtr_MPI_stats_t` structure containing the MPI statistics.
 * @param threadid The thread ID for which to reset the statistics data.
 */
void xtr_stats_MPI_reset(int threadid, xtr_MPI_stats_t * mpi_stats )
{
	int i;
  stats_mpi_thread_data_t * origin_data = xtr_get_mpi_stats_data(mpi_stats, threadid);

	if (TRACING_MPI_STATISTICS && origin_data != NULL)
	{
		origin_data->P2P_Bytes_Sent = 0;
		origin_data->P2P_Bytes_Recv = 0;
		origin_data->COLLECTIVE_Bytes_Sent = 0;
		origin_data->COLLECTIVE_Bytes_Recv = 0;
		origin_data->P2P_Communications = 0;
		origin_data->COLLECTIVE_Communications = 0;
		origin_data->MPI_Others_count = 0;

		origin_data->P2P_Communications_In = 0;
		origin_data->P2P_Communications_Out = 0;

		origin_data->begin_time[OTHER] = 0;
		origin_data->begin_time[P2P] = 0;
		origin_data->begin_time[COLLECTIVE] = 0;

		origin_data->elapsed_time[P2P] = 0;
		origin_data->elapsed_time[COLLECTIVE] = 0;

		for (i = 0; i < origin_data->ntasks; i++)
		{
			origin_data->P2P_Partner_In[i] = 0;
			origin_data->P2P_Partner_Out[i] = 0;
		}
	}
}

/**
 * @brief Duplicates the MPI statistics data structure.
 *
 * This function creates a new MPI statistics data structure and copies the data from the 
 * provided `mpi_stats` structure into the new one.
 *
 * @param mpi_stats Pointer to the source `xtr_MPI_stats_t` structure to duplicate.
 * @return Pointer to the newly created `xtr_MPI_stats_t` structure containing the duplicated data.
 *         Returns NULL if `TRACING_MPI_STATISTICS` is not enabled or if `mpi_stats` is NULL.
 */
xtr_MPI_stats_t * xtr_stats_MPI_dup(xtr_MPI_stats_t * mpi_stats)
{
	xtr_MPI_stats_t * new_mpi_stats = NULL;
	if (TRACING_MPI_STATISTICS && mpi_stats != NULL)
	{
		new_mpi_stats = xtr_mpi_stats_new();
		new_mpi_stats->num_threads = mpi_stats->num_threads;
		new_mpi_stats->common_stats_field.category = MPI_STATS_GROUP;
		for(int i=0; i < Backend_getMaximumOfThreads(); ++i)
		{
			xtr_stats_MPI_copy(i, mpi_stats, new_mpi_stats);
		}
	}
	return new_mpi_stats;
}

/**
 * @brief Copies the MPI statistics data from one structure to another for a specific thread.
 *
 * This function copies the MPI statistics from the `stats_origin` structure to the `stats_destination`
 * structure for the specified thread ID.
 *
 * @param stats_origin Pointer to the source `xtr_MPI_stats_t` structure.
 * @param stats_destination Pointer to the destination `xtr_MPI_stats_t` structure.
 * @param threadid The thread ID for which to copy the statistics data.
 */
void xtr_stats_MPI_copy(int threadid, xtr_MPI_stats_t * stats_origin, xtr_MPI_stats_t * stats_destination )
{
	if( threadid >= Backend_getMaximumOfThreads() )
		return;

  if (TRACING_MPI_STATISTICS)
  {
		stats_mpi_thread_data_t *origin_data = xtr_get_mpi_stats_data(stats_origin, threadid);
		stats_mpi_thread_data_t *destination_data = xtr_get_mpi_stats_data(stats_destination, threadid);
		if(origin_data != NULL && destination_data != NULL)
		{
			destination_data->P2P_Bytes_Sent = origin_data->P2P_Bytes_Sent;
			destination_data->P2P_Bytes_Recv = origin_data->P2P_Bytes_Recv;
			destination_data->COLLECTIVE_Bytes_Sent = origin_data->COLLECTIVE_Bytes_Sent;
			destination_data->COLLECTIVE_Bytes_Recv = origin_data->COLLECTIVE_Bytes_Recv;
			destination_data->P2P_Communications = origin_data->P2P_Communications;
			destination_data->COLLECTIVE_Communications = origin_data->COLLECTIVE_Communications;
			destination_data->MPI_Others_count = origin_data->MPI_Others_count;

			destination_data->P2P_Communications_In = origin_data->P2P_Communications_In;
			destination_data->P2P_Communications_Out = origin_data->P2P_Communications_Out;

			destination_data->begin_time[OTHER] = origin_data->begin_time[OTHER];
			destination_data->begin_time[P2P] = origin_data->begin_time[P2P];
			destination_data->begin_time[COLLECTIVE] = origin_data->begin_time[COLLECTIVE];

			destination_data->elapsed_time[OTHER] = origin_data->elapsed_time[OTHER];
			destination_data->elapsed_time[P2P] = origin_data->elapsed_time[P2P];
			destination_data->elapsed_time[COLLECTIVE] = origin_data->elapsed_time[COLLECTIVE];

			for(int i=0; i < Extrae_get_num_tasks(); ++i)
			{
				((stats_mpi_thread_data_t *)(stats_destination->common_stats_field.data))[threadid].P2P_Partner_In[i] = ((stats_mpi_thread_data_t *)(stats_origin->common_stats_field.data))[threadid].P2P_Partner_In[i];
				((stats_mpi_thread_data_t *)(stats_destination->common_stats_field.data))[threadid].P2P_Partner_Out[i] = ((stats_mpi_thread_data_t *)(stats_origin->common_stats_field.data))[threadid].P2P_Partner_Out[i];
			}
		}
  }
}

/**
 * @brief Retrieves the MPI statistics data for a specific thread.
 *
 * This function retrieves the statistics data for a specific thread from 'mpi_stats'.
 *
 * @param mpi_stats Pointer to the `xtr_MPI_stats_t` structure containing the MPI statistics.
 * @param threadid The thread ID for which to retrieve the statistics data.
 * @return Pointer to the `stats_mpi_thread_data` structure for the specified thread, or NULL if the input is invalid.
 */
stats_mpi_thread_data_t * xtr_get_mpi_stats_data( xtr_MPI_stats_t * mpi_stats, int threadid )
{
	if(mpi_stats != NULL)
	{
		if(mpi_stats->common_stats_field.category == MPI_STATS_GROUP)
		{
			return &(((stats_mpi_thread_data_t *)mpi_stats->common_stats_field.data)[threadid]);
		}
	}
	return NULL;
}

/**
 * @brief Subtracts MPI statistics of one data structure from another and stores the result in a destination structure.
 *
 * @param mpi_stats Pointer to the `xtr_MPI_stats_t` structure containing the original statistics.
 * @param subtrahend Pointer to the `xtr_MPI_stats_t` structure containing the statistics to subtract.
 * @param destination Pointer to the `xtr_MPI_stats_t` structure where the result will be stored.
 * @param threadid The thread ID for which the statistics are being subtracted.
 */
void xtr_stats_MPI_subtract (int threadid, xtr_MPI_stats_t *mpi_stats, xtr_MPI_stats_t *subtrahend, xtr_MPI_stats_t *destination)
{
	int i;
	unsigned diff;

	if(TRACING_MPI_STATISTICS)
	{
		stats_mpi_thread_data_t * origin_data = xtr_get_mpi_stats_data(mpi_stats, threadid);
		stats_mpi_thread_data_t * subtrahend_data = xtr_get_mpi_stats_data(subtrahend, threadid);
		stats_mpi_thread_data_t * destination_data = xtr_get_mpi_stats_data(destination, threadid);
		if(origin_data != NULL && subtrahend_data != NULL && destination_data != NULL)
		{
			destination_data->P2P_Bytes_Sent = origin_data->P2P_Bytes_Sent - subtrahend_data->P2P_Bytes_Sent;
			destination_data->P2P_Bytes_Recv = origin_data->P2P_Bytes_Recv - subtrahend_data->P2P_Bytes_Recv;
			destination_data->COLLECTIVE_Bytes_Sent = origin_data->COLLECTIVE_Bytes_Sent- subtrahend_data->COLLECTIVE_Bytes_Sent;
			destination_data->COLLECTIVE_Bytes_Recv = origin_data->COLLECTIVE_Bytes_Recv-subtrahend_data->COLLECTIVE_Bytes_Recv;
			destination_data->P2P_Communications = origin_data->P2P_Communications-subtrahend_data->P2P_Communications;
			destination_data->COLLECTIVE_Communications = origin_data->COLLECTIVE_Communications-subtrahend_data->COLLECTIVE_Communications;
			destination_data->MPI_Others_count = origin_data->MPI_Others_count- subtrahend_data->MPI_Others_count;

			destination_data->P2P_Communications_In = origin_data->P2P_Communications_In-subtrahend_data->P2P_Communications_In;
			destination_data->P2P_Communications_Out = origin_data->P2P_Communications_Out-subtrahend_data->P2P_Communications_Out;

			for (i =0; i< NUM_MPI_CATEGORIES; ++i)
			{
				destination_data->elapsed_time[i] = origin_data->elapsed_time[i] - subtrahend_data->elapsed_time[i];
			}

			for (i = 0; i < origin_data->ntasks; ++i)
			{
				diff = origin_data->P2P_Partner_In[i] - subtrahend_data->P2P_Partner_In[i];
				destination_data->P2P_Partner_In[i] = diff > 0 ? diff : 0;
				diff = origin_data->P2P_Partner_Out[i] - subtrahend_data->P2P_Partner_Out[i];
				destination_data->P2P_Partner_Out[i] = diff > 0 ? diff : 0;
			}
		}
	}
}

/**
 * @brief Retrieves MPI statistics for a given thread and stores them in provided arrays.
 *
 * @param mpi_stats Pointer to the `xtr_MPI_stats_t` structure containing the MPI statistics.
 * @param out_statistic_type Pointer to the array where the types of the statistics will be stored.
 * @param out_values Pointer to the array where the values of the statistics will be stored.
 * @param threadid The thread ID for which the statistics are being retrieved.
 * 
 * @return The number of statistics events recorded.
 */
int xtr_stats_MPI_get_values(int threadid, xtr_MPI_stats_t * mpi_stats, INT32 * out_statistic_type, UINT64 * out_values)
{
	int nevents = 0;
	int num_partners;
	unsigned long long elapsed_time_in_mpi;

	if (TRACING_MPI_STATISTICS)
	{
		stats_mpi_thread_data_t * thread_data = xtr_get_mpi_stats_data(mpi_stats, threadid);
		if( thread_data != NULL)
		{
			out_values[nevents] = thread_data->P2P_Communications;
			out_statistic_type[nevents++] = MPI_BURST_STATS_P2P_COUNT;
			out_values[nevents] = thread_data->P2P_Bytes_Sent;
			out_statistic_type[nevents++] =MPI_BURST_STATS_P2P_BYTES_SENT;
			out_values[nevents] = thread_data->P2P_Bytes_Recv;
			out_statistic_type[nevents++] =MPI_BURST_STATS_P2P_BYTES_RECV;
			out_values[nevents] = thread_data->COLLECTIVE_Communications;
			out_statistic_type[nevents++] =MPI_BURST_STATS_GLOBAL_COUNT;
			out_values[nevents] = thread_data->COLLECTIVE_Bytes_Sent;
			out_statistic_type[nevents++] =MPI_BURST_STATS_GLOBAL_BYTES_SENT;
			out_values[nevents] = thread_data->COLLECTIVE_Bytes_Recv;
			out_statistic_type[nevents++] =MPI_BURST_STATS_GLOBAL_BYTES_RECV;
			out_values[nevents] = thread_data->elapsed_time[COLLECTIVE];
			out_statistic_type[nevents++] =MPI_BURST_STATS_TIME_IN_GLOBAL;
			out_values[nevents] = thread_data->elapsed_time[P2P];
			out_statistic_type[nevents++] =MPI_BURST_STATS_TIME_IN_P2P;
			out_values[nevents] = thread_data->elapsed_time[OTHER];
			out_statistic_type[nevents++] =MPI_BURST_STATS_TIME_IN_OTHER;
			out_values[nevents] = thread_data->P2P_Communications_Out;
			out_statistic_type[nevents++] =MPI_BURST_STATS_P2P_OUTGOING_COUNT;
			out_values[nevents] = thread_data->P2P_Communications_In;
			out_statistic_type[nevents++] =MPI_BURST_STATS_P2P_INCOMING_COUNT;
			out_values[nevents] = thread_data->MPI_Others_count;
			out_statistic_type[nevents++] =MPI_BURST_STATS_OTHER_COUNT;

			elapsed_time_in_mpi = thread_data->elapsed_time[OTHER] + thread_data->elapsed_time[COLLECTIVE] + thread_data->elapsed_time[P2P];
			out_values[nevents] = elapsed_time_in_mpi;
			out_statistic_type[nevents++] = MPI_BURST_STATS_TIME_IN_MPI;

			num_partners = mpi_stats_get_num_partners(thread_data, thread_data->P2P_Partner_Out);
			out_values[nevents] = num_partners;
			out_statistic_type[nevents++] =MPI_BURST_STATS_P2P_OUTGOING_PARTNERS_COUNT;
			num_partners = mpi_stats_get_num_partners(thread_data, thread_data->P2P_Partner_In);
			out_values[nevents] = num_partners;
			out_statistic_type[nevents++] =MPI_BURST_STATS_P2P_INCOMING_PARTNERS_COUNT;
		}
	}
  return nevents;
}

/**
 * @brief Retrieves the statistics with positive values and their corresponding types for a specific thread.
 * 
 * @param mpi_stats Pointer to the MPI statistics object.
 * @param out_statistic_type Pointer to the array to store statistic types.
 * @param out_values Pointer to the array to store statistic values.
 * @param threadid The ID of the thread.
 * @return int The number of positive values retrieved.
 */
int xtr_stats_MPI_get_positive_values(int threadid, xtr_MPI_stats_t * mpi_stats, INT32 *out_statistic_type, UINT64 * out_values)
{
	int nevents = 0;
	int num_partners;
	unsigned long long elapsed_time_in_mpi;

	if (TRACING_MPI_STATISTICS && out_statistic_type != NULL && out_values != NULL)
	{
		stats_mpi_thread_data_t * thread_data = xtr_get_mpi_stats_data(mpi_stats, threadid);
		if(thread_data != NULL)
		{
			if ((out_values[nevents] = thread_data->elapsed_time[P2P]) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_TIME_IN_P2P;
			if ((out_values[nevents] = thread_data->elapsed_time[COLLECTIVE]) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_TIME_IN_GLOBAL;
			if ((out_values[nevents] = thread_data->elapsed_time[OTHER]) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_TIME_IN_OTHER;

			elapsed_time_in_mpi = thread_data->elapsed_time[OTHER] + thread_data->elapsed_time[COLLECTIVE] + thread_data->elapsed_time[P2P];
			if ( (out_values[nevents] = elapsed_time_in_mpi) > 0 )
				out_statistic_type[nevents++] = MPI_BURST_STATS_TIME_IN_MPI;
			if ((out_values[nevents] = thread_data->P2P_Bytes_Sent) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_P2P_BYTES_SENT;
			if ((out_values[nevents] = thread_data->P2P_Bytes_Recv) > 0)
				out_statistic_type[nevents++] =MPI_BURST_STATS_P2P_BYTES_RECV;
			if ((out_values[nevents] = thread_data->P2P_Communications) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_P2P_COUNT;
			if ((out_values[nevents] = thread_data->P2P_Communications_In) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_P2P_INCOMING_COUNT;
			if ((out_values[nevents] = thread_data->P2P_Communications_Out) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_P2P_OUTGOING_COUNT;
			if ((out_values[nevents] = thread_data->COLLECTIVE_Bytes_Sent) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_GLOBAL_BYTES_SENT;
			if ((out_values[nevents] = thread_data->COLLECTIVE_Bytes_Recv) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_GLOBAL_BYTES_RECV;
			if ((out_values[nevents] = thread_data->COLLECTIVE_Communications) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_GLOBAL_COUNT;
			if ((out_values[nevents] = thread_data->MPI_Others_count) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_OTHER_COUNT;

			num_partners = mpi_stats_get_num_partners(thread_data, thread_data->P2P_Partner_In);
			if ( (out_values[nevents] = num_partners) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_P2P_INCOMING_PARTNERS_COUNT;

			num_partners = mpi_stats_get_num_partners(thread_data, thread_data->P2P_Partner_Out);
			if ( (out_values[nevents] = num_partners) > 0)
				out_statistic_type[nevents++] = MPI_BURST_STATS_P2P_OUTGOING_PARTNERS_COUNT;
		}
	}
	return nevents;
}


/**
 * @brief Retrieves the description table with all the ids and their labels.
 * 
 */
stats_info_t *xtr_stats_MPI_get_types_and_descriptions(void)
{
  return &MPI_stats_info;
}

/**
 * UPDATE ROUTINES
 * These routines increase the statistics values of the different counters we keep track for MPI
 * These counters are divided into three categories: P2P, Collectives and Others.
*/

/**
 * @brief Updates MPI statistics regarding P2P communications
 */
void xtr_stats_MPI_update_P2P(UINT64 begin_time, UINT64 end_time, int partner, int inputSize, int outputSize)
{
	if (TRACING_MPI_STATISTICS)
	{
		stats_mpi_thread_data_t * thread_data = xtr_get_mpi_stats_data(MPI_stats, THREADID);
		if(thread_data != NULL)
		{
			thread_data->elapsed_time[P2P] += end_time-begin_time;
			/* Weird cases: MPI_Sendrecv_Fortran_Wrapper */
			thread_data->P2P_Communications ++;
			if (inputSize)
			{
					thread_data->P2P_Bytes_Recv += inputSize;
					thread_data->P2P_Communications_In ++;
					if (partner != MPI_PROC_NULL && partner != MPI_ANY_SOURCE && 
							partner != MPI_UNDEFINED)
					{
							if (partner < thread_data->ntasks)
									thread_data->P2P_Partner_In[partner] ++;
							// else
							// 		fprintf(stderr, "[DEBUG] OUT_OF_RANGE partner=%d/%d\n",
							// 				partner, thread_data->ntasks);
					}
			}
			if (outputSize)
			{
					thread_data->P2P_Bytes_Sent += outputSize;
					thread_data->P2P_Communications_Out ++;
					if (partner != MPI_PROC_NULL && partner != MPI_ANY_SOURCE && 
							partner != MPI_UNDEFINED)
					{
							if (partner < thread_data->ntasks)
									thread_data->P2P_Partner_Out[partner] ++;
							// else
							// 		fprintf(stderr, "[DEBUG] OUT_OF_RANGE partner=%d/%d\n",
							// 				partner, thread_data->ntasks);
					}
			}
			DEBUG_MSG("[TID:%d THD:%d] updated MPI P2P statistics. inputSize %d outputSize %d. acumulated = %d",
					TASKID, THREADID, inputSize, outputSize, thread_data->P2P_Communications);
		}
	}
}

/**
 * @brief Updates MPI statistics regarding Collective communications
 */
void xtr_stats_MPI_update_collective(UINT64 begin_time, UINT64 end_time, int inputSize, int outputSize)
{

	if (TRACING_MPI_STATISTICS)
	{
		stats_mpi_thread_data_t * thread_data = xtr_get_mpi_stats_data(MPI_stats, THREADID);
		if(thread_data != NULL)
		{
			thread_data->elapsed_time[COLLECTIVE] += end_time-begin_time;

			thread_data->COLLECTIVE_Communications ++;
			if (inputSize)
			{
					thread_data->COLLECTIVE_Bytes_Recv += inputSize;
			}
			if (outputSize)
			{
				thread_data->COLLECTIVE_Bytes_Sent += outputSize;
			}
			DEBUG_MSG("[TID:%d THD:%d] updated MPI COLLECTIVE statistics. inputSize %d outputSize %d. aculumated %d",
				TASKID, THREADID, inputSize, outputSize, thread_data->COLLECTIVE_Communications);
		}
	}
}

/**
 * @brief Updates MPI statistics of the remaining runtime calls
 */
void xtr_stats_MPI_update_other(UINT64 begin_time, UINT64 end_time )
{
	if (TRACING_MPI_STATISTICS)
	{
		stats_mpi_thread_data_t * thread_data = xtr_get_mpi_stats_data(MPI_stats, THREADID);
		if(thread_data != NULL)
		{
			thread_data->elapsed_time[OTHER] += end_time-begin_time;

			thread_data->MPI_Others_count++;
			DEBUG_MSG("[TID:%d THD:%d] updated MPI OTHER statistics, acumulated  = %d ", 
				TASKID, THREADID, thread_data->MPI_Others_count);
		}
	}
}

/**
 * @brief returns the number of partners present in a given statistic data object
 */
int mpi_stats_get_num_partners(stats_mpi_thread_data_t * data, int * partners_vector)
{
  int i, num_partners = 0;
  for (i = 0; i < data->ntasks; i++)
  {
    if (partners_vector[i] > 0) num_partners++;
  }
  return num_partners;
}

/**
 * @brief Frees the memory allocated for MPI statistics.
 *
 * @param mpi_stats Pointer to the `xtr_MPI_stats_t` structure to be freed.
 */
void xtr_stats_MPI_free (xtr_MPI_stats_t * mpi_stats )
{
	for(int i=0; i < mpi_stats->num_threads ; ++i)
	{
		xfree(((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_In);
		xfree(((stats_mpi_thread_data_t *)(mpi_stats->common_stats_field.data))[i].P2P_Partner_Out);
	}

	xfree(mpi_stats->common_stats_field.data);
}

void xtr_print_debug_mpi_stats ( int threadid )
{
	if(MPI_stats == NULL)
		return;
  fprintf (stderr, PACKAGE_NAME " [DEBUG] [TID:%d THD:%d] MPI_stats: \n \
            P2P_Bytes_Sent: %d \n \
            P2P_Bytes_Recv: %d \n \
            COLLECTIVE_Bytes_Sent: %d \n \
            COLLECTIVE_Bytes_Recv: %d \n \
            P2P_Communications: %d \n \
            COLLECTIVE_Communications: %d \n \
            MPI_Others_count: %d \n \
            P2P_Communications_In: %d \n \
            P2P_Communications_Out: %d \n \
            elapsed time in OTHER: %d \n \
            elapsed time in P2P: %d \n \
            elapsed time in COLLECTIVE: %d \n",
          TASKID, threadid, ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].P2P_Bytes_Sent,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].P2P_Bytes_Recv,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].COLLECTIVE_Bytes_Sent,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].COLLECTIVE_Bytes_Recv,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].P2P_Communications,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].COLLECTIVE_Communications,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].MPI_Others_count,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].P2P_Communications_In,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].P2P_Communications_Out,
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].elapsed_time[OTHER],
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].elapsed_time[P2P],
          ((stats_mpi_thread_data_t *)MPI_stats->common_stats_field.data)[threadid].elapsed_time[COLLECTIVE]
          );
}
