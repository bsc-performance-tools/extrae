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

#include "common.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif
#include "intercommunicators.h"
#include "common.h"
#include "utils.h"

ptask_to_spawn_group_t *AppToSpawnGroupTable = NULL;
int num_SpawnGroups = 0;

spawn_group_table_t *IntercommTable = NULL;


void intercommunicators_load( char *spawns_file_path, int ptask )
{
  char *spawns_file     = strdup( spawns_file_path );
  char *spawn_group_str = NULL;
  int   SpawnGroup;

  /* Remove the path */
  spawns_file = basename( spawns_file ); 
  
  /* Remove the extension */
  spawns_file[strlen(spawns_file)-strlen(EXT_SPAWN)] = '\0';

  /* Parse the spawn group id */
  spawn_group_str = rindex( spawns_file, '-' );

  if ((spawn_group_str == NULL) || (strlen(spawn_group_str) == 0))
  {
    SpawnGroup = 1;
  }
  else
  {
    spawn_group_str ++;
    SpawnGroup = atoi(spawn_group_str);
  }

  intercommunicators_map_ptask_to_spawn_group( SpawnGroup, ptask );

  intercommunicators_allocate_links( SpawnGroup );

  /* Parse the links in the file */
  FILE *fd = fopen(spawns_file_path, "r");
  char line[256];

  fgets(line, sizeof(line), fd); /* Skip the first line (the synchronization latency) */
  while (fgets(line, sizeof(line), fd)) 
  {
    int from_task, from_comm, to_spawn_group;

    sscanf(line, "%d %d %d", &from_task, &from_comm, &to_spawn_group);

    intercommunicators_new_link(SpawnGroup, from_task, from_comm, to_spawn_group);
  }
  fclose(fd);

  /* DEBUG 
  intercommunicators_print(); */
}

void intercommunicators_map_ptask_to_spawn_group( int SpawnGroup, int ptask )
{
  /* Store the translation between ptask and spawn group */
  xrealloc(AppToSpawnGroupTable, AppToSpawnGroupTable, (num_SpawnGroups+1) * sizeof(ptask_to_spawn_group_t));
  AppToSpawnGroupTable[num_SpawnGroups].ptask = ptask;
  AppToSpawnGroupTable[num_SpawnGroups].spawn_group = SpawnGroup;
  num_SpawnGroups ++;
}

void intercommunicators_allocate_links( int SpawnGroup )
{
  int i;

  /* Allocate room for storing the links of this spawn group */
  if (IntercommTable == NULL)
  {
    IntercommTable = (spawn_group_table_t *)malloc(sizeof(spawn_group_table_t));
    IntercommTable->groups     = NULL;
    IntercommTable->num_groups = 0;
  }
  if (SpawnGroup > IntercommTable->num_groups)
  {
    for (i=IntercommTable->num_groups; i<SpawnGroup; i++)
    {
      xrealloc(IntercommTable->groups, IntercommTable->groups, SpawnGroup * sizeof(spawn_group_t));

      IntercommTable->groups[ i ].num_links = 0;
      IntercommTable->groups[ i ].links     = NULL;
    }
    IntercommTable->num_groups = SpawnGroup;
  }
}

void intercommunicators_new_link(int from_spawn_group, int from_task, int from_comm, int to_spawn_group)
{
  spawn_group_t *group = &(IntercommTable->groups[from_spawn_group - 1]);

  xrealloc(group->links, group->links, (group->num_links+1) * sizeof(link_t));

  group->links[ group->num_links ].from_task = from_task;  
  group->links[ group->num_links ].from_comm = from_comm;  
  group->links[ group->num_links ].to_spawn_group = to_spawn_group;  

  group->num_links ++;
}

void intercommunicators_print()
{
  int i, j;

  if (IntercommTable != NULL)
  {
    fprintf(stderr, "intercommunicators_print: Dumping %d spawn groups...\n", IntercommTable->num_groups);
    for (i=0; i<IntercommTable->num_groups; i++)
    {
      fprintf(stderr, "intercommunicators_print: Links for spawn group %d\n", i+1);
      for (j=0; j<IntercommTable->groups[i].num_links; j++)
      {
        fprintf(stderr, "link #%d: from_task=%d from_comm=%d to_spawn_group=%d\n", j+1,
          IntercommTable->groups[i].links[j].from_task,
          IntercommTable->groups[i].links[j].from_comm,
          IntercommTable->groups[i].links[j].to_spawn_group);
      }
    }
  }

  for (i=0; i<num_SpawnGroups; i++)
  {
    fprintf(stderr, "PTASK %d -> SPAWN_GROUP %d\n", 
      AppToSpawnGroupTable[i].ptask,
      AppToSpawnGroupTable[i].spawn_group);
  }
}


static int get_spawn_group( int ptask )
{
  int i;

  for (i=0; i<num_SpawnGroups; i++)
  {
    if (AppToSpawnGroupTable[i].ptask == ptask)
      return AppToSpawnGroupTable[i].spawn_group;
  }
  return -1;
}

static int get_ptask( int spawn_group )
{
  int i;

  for (i=0; i<num_SpawnGroups; i++)
  {
    if (AppToSpawnGroupTable[i].spawn_group == spawn_group)
      return AppToSpawnGroupTable[i].ptask;
  }
  return -1;
}

static int find_link(int from_spawn_group, int from_task, int from_comm)
{
  int i;
  spawn_group_t *group = NULL;

  if (IntercommTable->num_groups > 0)
  {
    group = &(IntercommTable->groups[from_spawn_group-1]);

    for (i=0; i<group->num_links; i++)
    {
      if ((group->links[i].from_task == from_task-1) &&
          (group->links[i].from_comm == from_comm))
      {
        return group->links[i].to_spawn_group;
      }
    }
  }
  return -1;
}

int intercommunicators_get_target_ptask(int from_ptask, int from_task, int from_comm)
{
  int from_spawn_group = get_spawn_group(from_ptask);
  int to_spawn_group;
  int ret;

  if (from_spawn_group == -1)
  {
    /* There's no spawns! */
    ret = from_ptask;
  }
  else 
  { 
    to_spawn_group = find_link(from_spawn_group, from_task, from_comm); 

    if (to_spawn_group == -1)
    {
      /* This is not an intercommunicator, the target of the message is in the same ptask */
      ret = from_ptask;
    }
    else
    {
      /* This is an intercommunicator, find the target spawn group and translate to the corresponding ptask */
      int to_ptask = get_ptask( to_spawn_group );
      if (to_ptask == -1)
        ret = from_ptask;
      else
        ret = to_ptask; 
    }
  }
  /* DEBUG 
  fprintf(stderr, "\n[DEBUG] intercommunicators_get_target_ptask from_ptask=%d from_task=%d from_comm=%d target_ptask=%d\n",
    from_ptask, from_task, from_comm, ret); */
  return ret;
}

