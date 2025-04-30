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
#include "stats_module.h"

#include "common.h"
#include "change_mode.h"
#include "omp_stats.h"
#include "threadid.h"
#include "xalloc.h"

/**
 * This files manages the runtime statistics reported in burst mode.
 * To add a new runtime statistic provide the functions listed in "stats_vtable_st (stas_module.h)",
 *   define a new group in 'enum stats_group'(stas_module.h) and add another entry in "stats_vtable_st virtual_table" below.
 * 
 * This module stores the statistics of all the runtimes in 'RuntimeStats'
 * And their descriptions in 'RuntimeDescriptions' this is a type and a label for each statistic
 * 
 * Most of the rotines requiere an array of statistic objects as a parameter 
 * and a "count" argument indicating the number of elements in the array:
 * xtr_stats_t **stats_obj
 * This is an array of runtime statistics. each element of this array
 * stores the statistics of a single runtime (see definition in header file)
*/

/**
 * This variable stores the acumulation of all the runtime statistics
 * since the begining of the execution
*/
xtr_stats_t *RuntimeStats[NUM_STATS_GROUPS] = {0};
stats_info_t *RuntimeDescriptions[NUM_STATS_GROUPS] = {0};


/*
  virtual table to achieve polymorphisim
*/
stats_vtable_st virtual_table [NUM_STATS_GROUPS] = 
{
#if defined(NEW_OMP_SUPPORT)
	[OMP_STATS_GROUP] =
	{
		.reset = xtr_stats_OMP_reset,
		.copyto = xtr_stats_OMP_copy,
		.dup = xtr_stats_OMP_dup,
		.subtract = xtr_stats_OMP_subtract,
		.get_positive_values_and_ids = xtr_stats_OMP_get_positive_values,
		.get_ids_and_descriptions = xtr_stats_OMP_get_types_and_descriptions,
		.realloc = xtr_stats_OMP_realloc,
		.free = xtr_stats_OMP_free,
		.nevents = OMP_BURST_STATS_COUNT,
	},
#endif
#if defined(MPI_SUPPORT)
	[MPI_STATS_GROUP] =
	{
		.reset = xtr_stats_MPI_reset,
		.copyto = xtr_stats_MPI_copy,
		.dup = xtr_stats_MPI_dup,
		.subtract = xtr_stats_MPI_subtract,
		.get_positive_values_and_ids = xtr_stats_MPI_get_positive_values,
		.get_ids_and_descriptions = xtr_stats_MPI_get_types_and_descriptions,
		.realloc = xtr_stats_MPI_realloc,
		.free = xtr_stats_MPI_free,
		.nevents = MPI_BURST_STATS_COUNT,
	}
#endif
};

/**
 * @brief Initializes runtime statistics for OpenMP and MPI if they are enabled and not already initialized.
 * 
 * This function checks if tracing of OpenMP and MPI statistics is enabled. If so, and if the respective
 * runtime statistics are not already initialized, it initializes them.
 *
 * @return xtr_stats_st** Pointer to the array of runtime statistics objects. The arraay will have a NULL in the place of a disabled runtime
 *
 * @note The caller must ensure that the return value is handled appropriately
 */
xtr_stats_t ** xtr_stats_initialize( void )
{
#if defined(NEW_OMP_SUPPORT)
	if (TRACING_OMP_STATISTICS && RuntimeStats[OMP_STATS_GROUP] == NULL)
	{
		RuntimeStats[OMP_STATS_GROUP] = xtr_stats_OMP_init();
		RuntimeDescriptions[OMP_STATS_GROUP] =  virtual_table[OMP_STATS_GROUP].get_ids_and_descriptions();
	}
#endif
#if defined(MPI_SUPPORT)
	if (TRACING_MPI_STATISTICS && RuntimeStats[MPI_STATS_GROUP] == NULL)
	{
		RuntimeStats[MPI_STATS_GROUP] = xtr_stats_MPI_init();
		RuntimeDescriptions[MPI_STATS_GROUP] = virtual_table[MPI_STATS_GROUP].get_ids_and_descriptions();
	}
#endif

	return RuntimeStats;
}

/**
 * @brief Reallocates resources for an array of statistics objects for a new number of threads.
 * 
 * This function reallocates the resources of each statistics object in the provided array
 * to accommodate a new number of threads.
 *
 * @param stats Pointer to the array of statistics objects to be reallocated.
 * @param new_num_threads The new number of threads to allocate resources for.
 *
 * @note The caller must ensure that `stats` is a valid pointer to an array of statistics objects.
 */
void xtr_stats_realloc (xtr_stats_t **stats, int old_num_threads, int new_num_threads)
{
	if(new_num_threads > old_num_threads)
	{
		for(int i =0; i<NUM_STATS_GROUPS; ++i)
		{
			if(stats[i] != NULL)
			{
				virtual_table[stats[i]->category].realloc( stats[i], new_num_threads );
			}
		}
	} 
}

/**
 * @brief reallocates the statistic module resources for a new number of threads.
 * 
 * This function reallocates resources for the runtime statistics objects to accommodate a new number of threads.
 *
 * @param new_num_threads The new number of threads to allocate resources for.
 * @note This operation is not thread safe
 */
void xtr_stats_change_nthreads(int old_num_threads, int new_num_threads)
{
	xtr_stats_realloc(RuntimeStats, old_num_threads, new_num_threads);
}

/**
 * @brief Resets an array of statistics objects.
 * 
 * This function sets every statistics field to zero in the given object.
 *
 * @param stats_obj Pointer to the array of statistics objects to be reset.
 *
 * @note The caller must ensure that `stats_obj` is a valid pointer to an array of statistics objects and that `count` is the correct number of elements in the array.
 */
void xtr_stats_reset(xtr_stats_t **stats_obj)
{
	for (int i=0; i<NUM_STATS_GROUPS; ++i)
	{
		if (stats_obj[i] != NULL)
			virtual_table[stats_obj[i]->category].reset(THREADID, stats_obj[i]);
	}
}

/**
 * @brief Duplicates an array of statistics objects.
 * 
 * This function creates a duplicate of an array of statistics objects, copying
 * each non-null statistics object using the appropriate duplication function
 * from a virtual table.
 *
 * @param stats_obj Pointer to the array of statistics objects to duplicate.
 * @return xtr_stats_st** A pointer to the newly duplicated array of statistics objects.
 *
 * @note The caller is responsible for freeing the memory allocated for the new array of statistics 
 * objects by calling * xtr_stats_free
 */
xtr_stats_t **xtr_stats_dup(xtr_stats_t **stats_obj)
{
	xtr_stats_t **new_obj = xmalloc_and_zero(sizeof(xtr_stats_t *) * NUM_STATS_GROUPS);

	for (int i=0; i<NUM_STATS_GROUPS; ++i)
	{
		if (stats_obj[i] != NULL)
		{
			new_obj[stats_obj[i]->category] = virtual_table[stats_obj[i]->category].dup(stats_obj[i]);
		}
	}

	return new_obj;
}

/**
 * @brief Copies statistics data from one array of statistics objects to another.
 * 
 * This function copies the data from each statistics object in the source array (`stats_obj_from`)
 * to the corresponding statistics object in the destination array (`stats_obj_to`).
 * The copy operation is performed only if both the source and destination statistics objects are non-null
 * and belong to the same category.
 *
 * @param stats_obj_from Pointer to the array of source statistics objects.
 * @param stats_obj_to Pointer to the array of destination statistics objects.
 *
 * @note The caller must ensure that `stats_obj_from` and `stats_obj_to` are valid pointers to arrays of statistics objects and that `count` is the correct number of elements in the arrays.
 */
void xtr_stats_copyto(xtr_stats_t **stats_obj_from, xtr_stats_t **stats_obj_to)
{
	for (int i=0; i<NUM_STATS_GROUPS; ++i)
	{
		if (stats_obj_from[i] != NULL && stats_obj_to[i] != NULL && stats_obj_from[i]->category == stats_obj_to[i]->category)
		{
			virtual_table[stats_obj_from[i]->category].copyto(THREADID, stats_obj_from[i], stats_obj_to[i]);
		}
	}
}

/**
 * @brief Retrieves the category of a given statistics object.
 * 
 * This function returns the category of the specified statistics object. If the
 * statistics object is `NULL`, it returns `WRONG_STATS_GROUP`.
 *
 * @param stats_obj Pointer to the statistics object whose category is to be retrieved.
 * @return int The category of the statistics object (MPI_STATS_GROUP,
 * OMP_STATS_GROUP), or `WRONG_STATS_GROUP` if the object is `NULL`.
 */
int xtr_stats_get_category(xtr_stats_t * stats_obj)
{
  return ((stats_obj != NULL ) ? stats_obj->category : WRONG_STATS_GROUP);
}

/**
 * @brief Retrieves positive values statistics and their types from an array of statistics objects.
 * 
 * This function iterates through an array of statistics objects, retrieves positive values and their corresponding types,
 * and stores them in the provided output arrays. The total number of positive values retrieved is stored in `out_size`.
 *
 * @param count The number of statistics objects in the array.
 * @param stats_obj Pointer to the array of statistics objects.
 * @param out_ids Pointer to the array where the types of positive values will be stored.
 * @param out_value Pointer to the array where the positive values will be stored.
 * @param out_size Pointer to an integer where the total number of positive values will be stored.
 * @param threadid The ID of the current thread to whom retrieve and perform the operation.
 *
 * @return int The total number of positive values retrieved.
 * 
 * @note Reports only positive values to avoid constantly dumping events with value 0 in spawn threads where no MPI is executed
 * @note The caller must ensure that `stats_obj`, `out_ids`, and `out_value` are valid pointers and that `count` is the correct number of elements in the `stats_obj` array.
 */
void xtr_stats_get_values(int threadid, xtr_stats_t **stats_obj, int out_num[], INT32 out_ids[][STATS_SIZE_PER_GROUP], UINT64 out_values[][STATS_SIZE_PER_GROUP])
{
	for (int i=0; i<NUM_STATS_GROUPS; ++i)
	{
		if (stats_obj[i] != NULL)
		{
			out_num[i] = virtual_table[stats_obj[i]->category].get_positive_values_and_ids(threadid, stats_obj[i], out_ids[i], out_values[i]);
		}
	}
}

/**
 * @brief Retrieves the statistics ids and descriptions for all loaded statistics objects.
 * 
 * This function retrieves the IDs and descriptions for the enabled runtime statistics objects
 * Returning the addess of the array that holds them 'RuntimeDescriptions'
 * Each position on the array corresponds to a specific runtime object description
 * see (stats_types.h) ,'OMP_stats_info' and 'MPI_stats_info' in 'omp_stats.c' and 'mpi_stats.c'
 * both of them have a terminating element that has the values {-1, NULL}
 * The 'stats_info_t' struct that stores a pair of elements:
 * id and description.
 *
 * @return The Address of the description table of the runtime statistics
 * 
 */
stats_info_t **xtr_stats_get_description_table(void)
{
	return RuntimeDescriptions;
}

/**
 * @brief Subtracts one array of statistics objects from another and stores the result in a destination array.
 * 
 * This function performs element-wise subtraction of statistics objects in the `subtrahend` array from the corresponding
 * statistics objects in the `minuend` array, and stores the result in the `destination` array.
 * The subtraction operation is performed only if all three statistics objects (minuend, subtrahend, and destination) are non-null
 * and belong to the same category.
 *
 * @param count The number of statistics objects in the arrays.
 * @param minuend Pointer to the array of minuend statistics objects (from which values are subtracted).
 * @param subtrahend Pointer to the array of subtrahend statistics objects (which are subtracted).
 * @param destination Pointer to the array where the results of the subtraction are stored.
 * @param threadid The ID of the thread to which retrieve its statistics and perform the operation.
 *
 * The function performs the following operations:
 * - Iterates over each statistics object up to the specified count.
 * - For each set of minuend, subtrahend, and destination statistics objects:
 *   - Checks if all three statistics objects are non-null.
 *   - Checks if the categories of the minuend and subtrahend statistics objects match.
 *   - Uses the category to select the appropriate subtraction function from the virtual table.
 *   - Calls the subtraction function with the minuend, subtrahend, and destination statistics objects and the specified thread ID.
 *
 * @note The routine is not thread safe
 * @note The caller must ensure that `minuend`, `subtrahend`, and `destination` are valid pointers to arrays of statistics objects and that `count` is the correct number of elements in the arrays.
 */
void xtr_stats_subtract (int threadid, xtr_stats_t **minuend, xtr_stats_t **subtrahend, xtr_stats_t **destination)
{
	for (int i = 0; i< NUM_STATS_GROUPS; ++i)
	{
		if(minuend[i] != NULL && subtrahend[i] != NULL && destination[i] != NULL && minuend[i]->category == subtrahend[i]->category)
			virtual_table[minuend[i]->category].subtract(threadid, minuend[i], subtrahend[i], destination[i]);
	}
}

/**
 * @brief Frees memory allocated for an array of statistics objects.
 * 
 * This function deallocates memory for each statistics object in the provided array.
 *
 * @param count The number of statistics objects in the array.
 * @param stats Pointer to the array of statistics objects to be freed.
 * 
 * @note The caller must ensure that the statistics objects are no longer needed before calling this function.
 */
void xtr_stats_free (xtr_stats_t **stats)
{
	for (int i =0; i<NUM_STATS_GROUPS; ++i)
	{
		if(stats[i] != NULL) virtual_table[stats[i]->category].free ( stats[i] );
	}
}

/**
 * @brief Finalizes and releases resources allocated for runtime statistics module
 * 
 * This function deallocates memory and finalizes the runtime statistics objects.
 *
 * @note The caller must ensure that the runtime statistics objects are no longer needed before calling this function.
 */
void xtr_stats_finalize()
{
	xtr_stats_free(RuntimeStats);
}

/**
 * @brief Prints debug statistics for the specified thread.
 * 
 * This function prints debug statistics for the specified thread if tracing of MPI or OpenMP statistics is enabled.
 *
 * @param tid The ID of the thread for which debug statistics are printed.
 */
void xtr_print_debug_stats ( int tid )
{
#if defined(MPI_SUPPORT)
	if (TRACING_MPI_STATISTICS)
  		xtr_print_debug_mpi_stats(tid);
#endif
#if defined(NEW_OMP_SUPPORT)
	if (TRACING_OMP_STATISTICS)
	  xtr_print_debug_omp_stats(tid);
#endif
}
