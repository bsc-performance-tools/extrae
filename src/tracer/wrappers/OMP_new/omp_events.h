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

#include "events.h"

void Extrae_OpenMP_Counters (void);
void Extrae_OpenMP_Call_Entry (unsigned omp_call);
void Extrae_OpenMP_Call_Exit (unsigned omp_call);
void Extrae_OpenMP_Parallel_Entry (unsigned par_construct);
void Extrae_OpenMP_Parallel_Exit ();
void Extrae_OpenMP_Forking_Entry (unsigned par_construct);
void Extrae_OpenMP_Forking_Exit ();
void Extrae_OpenMP_Chunk_Entry (void);
void Extrae_OpenMP_Chunk_Exit (void);
void Extrae_OpenMP_Worksharing_Entry (unsigned wsh_construct);
void Extrae_OpenMP_Worksharing_Exit ();
void Extrae_OpenMP_Sync_Entry (unsigned sync_construct);
void Extrae_OpenMP_Sync_Exit (void);
void Extrae_OpenMP_Lock_Status (const void *name, unsigned lock_state);
void Extrae_OpenMP_Ordered (unsigned int ordered_state);
void Extrae_OpenMP_Taskgroup (unsigned int taskgroup_state);
void Extrae_OpenMP_Outlined_Entry (const void *outlined_fn);
void Extrae_OpenMP_Outlined_Exit (void);
void Extrae_OpenMP_Outlined_Entry_At (UINT64 time, const void *outlined_fn);
void Extrae_OpenMP_Outlined_Exit_At ( UINT64 time );
void Extrae_OpenMP_Task_Inst_Entry (const void *task, long long task_id);
void Extrae_OpenMP_Task_Inst_Exit (void);
void Extrae_OpenMP_Task_Exec_Entry (int task_or_taskloop, const void *task, long long task_id);
void Extrae_OpenMP_Task_Exec_Exit (void);
void Extrae_OpenMP_Taskloop_Entry (const void *task, long long taskloop_id, int num_tasks);
void Extrae_OpenMP_Taskloop_Exit (void);
void Extrae_OpenMP_Target_Entry (unsigned);
void Extrae_OpenMP_Target_Exit (void);
