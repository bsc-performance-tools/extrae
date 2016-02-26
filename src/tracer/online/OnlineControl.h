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

#ifndef __ONLINE_CONTROL_H__
#define __ONLINE_CONTROL_H__

#include "OnlineConfig.h"

//#define MASTER_BACKEND_RANK(world_size) (world_size - 1) /* Last MPI process runs the master back-end by default */
#define MASTER_BACKEND_RANK(world_size) 0

/**
 * Structure to pass data to the back-end thread
 */
typedef struct
{
  int  my_rank;
  char parent_hostname[128];
  int  parent_port;
  int  parent_rank;
} BE_data_t;

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
int  Online_Init(int rank, int world_size);
int  Online_Start(char **node_list);
void Online_Stop(void);
void Online_CleanTemporaries(void);
#if defined(__cplusplus)
}
#endif /* __cplusplus */

#if defined(__cplusplus)
void Online_PauseApp(bool emit_events = true);
void Online_ResumeApp(bool emit_events = true);
unsigned long long Online_GetAppPauseTime();
unsigned long long Online_GetAppResumeTime();

void Online_Flush(void);
char * Online_GetTmpBufferName(void);
char * Online_GetFinalBufferName(void);

void * BE_main_loop(void *context);
#endif /* __cplusplus */

#endif /* __ONLINE_CONTROL_H__ */
