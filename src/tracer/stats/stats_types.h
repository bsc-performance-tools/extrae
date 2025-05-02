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

#pragma once

/**
 * runtime groups
 */
enum stats_group
{
  WRONG_STATS_GROUP = -1,
  MPI_STATS_GROUP,
  OMP_STATS_GROUP,
  NUM_STATS_GROUPS
};

/**
 * Base statistics structure.
 * 
 * This structure must be the first field in any runtime-specific
 * statistics structure. This design allows access to the common
 * fields defined here without requiring explicit casting.
 * 
 * A statistics object serves two main purposes:
 * 1. To store the accumulated statistics up to a given point in time.
 * 2. To store the differences (deltas) between two statistics objects 
 *    captured at different timestamps.
 */
typedef struct xtr_stats
{
  enum stats_group category; // Indicates the runtime category/type
  void *data;                // Pointer to runtime-specific statistics data
} xtr_stats_t;


/**
 * Used to create a table with all the statistics ids of a runtime and their descriptions
 */
typedef struct stats_info{
  int id;
  char *description;
}stats_info_t;