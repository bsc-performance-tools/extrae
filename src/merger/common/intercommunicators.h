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

#ifndef __INTERCOMMUNICATORS_H__
#define __INTERCOMMUNICATORS_H__

#include "mpi2out.h"

typedef struct
{
  int ptask;
  int spawn_group;
} ptask_to_spawn_group_t;

typedef struct
{
  int from_task;
  int from_comm;
  int to_spawn_group;
} link_t;

typedef struct
{
  int num_links;
  link_t *links;
} spawn_group_t;

typedef struct
{
  spawn_group_t *groups;
  int            num_groups;
} spawn_group_table_t;

void intercommunicators_load(char *spawns_file_path, int ptask);
void intercommunicators_map_ptask_to_spawn_group( int SpawnGroup, int ptask );
void intercommunicators_allocate_links( int SpawnGroup );
void intercommunicators_new_link(int from_spawn_group, int from_task, int from_comm, int to_spawn_group);
void intercommunicators_print();
int intercommunicators_get_target_ptask(int from_ptask, int from_task, int from_comm);

#endif /* __INTERCOMMUNICATORS_H__ */
