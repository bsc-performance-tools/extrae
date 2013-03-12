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

#ifndef __ONLINE_CONTROL_H__
#define __ONLINE_CONTROL_H__

#include "OnlineConfig.h"

//#define ONLINE_DEBUG                               /* Define this to activate debug messages         */

#if defined(ONLINE_DEBUG)
# define ONLINE_DBG(msg, args...)                         \
{                                                         \
   fprintf(stderr, "[ONLINE %d%s] "                       \
     msg, this_BE_rank, (I_am_root ? "R" : ""), ## args); \
   fflush(stderr);                                        \
}
#else
# define ONLINE_DBG(msg, args...) { ; }
#endif

#define ONLINE_DBG_1 \
  if (I_am_root) ONLINE_DBG

#define FRONTEND_RANK(world_size) (world_size - 1) /* Last MPI process runs the front-end by default */

#define DEFAULT_FANOUT 32                          /* Default fan-out for the MRNet-tree             */

/**
 * Structure to pass data to the front-end thread
 */
typedef struct 
{
  char resources_file[128];
  char topology_file[128];
  int  num_backends;
  char attach_file[128];
} FE_thread_data_t;

/**
 * Structure to pass data to the back-end thread
 */
typedef struct
{
  int  my_rank;
  char parent_hostname[128];
  int  parent_port;
  int  parent_rank;
} BE_thread_data_t;


#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

int  Generate_Topology(int world_size, char **node_list, char *resources_file, char *topology_file);

int  Online_Start(int rank, int world_size, char **node_list);
void Online_Stop();
void Online_PauseApp();
void Online_ResumeApp();
unsigned long long Online_GetAppPauseTime();
unsigned long long Online_GetAppResumeTime();

void Online_Flush();
void Online_CleanTemporaries();
char * Online_GetTmpBufferName();
char * Online_GetFinalBufferName();

void * FE_main_loop(void *context);
void * BE_main_loop(void *context);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* __ONLINE_CONTROL_H__ */
