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

#ifndef OMP_PROBE_H_INCLUDED
#define OMP_PROBE_H_INCLUDED

void setTrace_OMPLocks (int value);
int getTrace_OMPLocks (void);

void setTrace_OMPTaskloop (int value);
int getTrace_OMPTaskloop (void);

void Probe_OpenMP_UF_Entry (UINT64 uf);
void Probe_OpenMP_UF_Exit (void);
void Probe_OpenMP_Work_Entry (void);
void Probe_OpenMP_Work_Exit (void);

void Probe_OpenMP_Join_NoWait_Entry (void);
void Probe_OpenMP_Join_NoWait_Exit (void);
void Probe_OpenMP_Join_Wait_Entry (void);
void Probe_OpenMP_Join_Wait_Exit (void);

void Probe_OpenMP_DO_Entry (void);
void Probe_OpenMP_DO_Exit (void);
void Probe_OpenMP_Sections_Entry (void);
void Probe_OpenMP_Sections_Exit (void);
void Probe_OpenMP_Single_Entry (void);
void Probe_OpenMP_Single_Exit (void);
void Probe_OpenMP_Master_Entry (void);
void Probe_OpenMP_Master_Exit (void);
void Probe_OpenMP_Section_Entry(void);
void Probe_OpenMP_Section_Exit (void);

void Probe_OpenMP_ParRegion_Entry (void);
void Probe_OpenMP_ParRegion_Exit (void);
void Probe_OpenMP_ParDO_Entry (void);
void Probe_OpenMP_ParDO_Exit (void);
void Probe_OpenMP_ParSections_Entry (void);
void Probe_OpenMP_ParSections_Exit (void);

void Probe_OpenMP_Barrier_Entry (void);
void Probe_OpenMP_Barrier_Exit (void);

void Probe_OpenMP_Named_Lock_Entry (void);
void Probe_OpenMP_Named_Lock_Exit (const void *name);
void Probe_OpenMP_Named_Unlock_Entry (const void *name);
void Probe_OpenMP_Named_Unlock_Exit (void);
void Probe_OpenMP_Unnamed_Lock_Entry (void);
void Probe_OpenMP_Unnamed_Lock_Exit (void);
void Probe_OpenMP_Unnamed_Unlock_Entry (void);
void Probe_OpenMP_Unnamed_Unlock_Exit (void);

void Probe_OpenMP_GetNumThreads_Entry (void);
void Probe_OpenMP_GetNumThreads_Exit (void);
void Probe_OpenMP_SetNumThreads_Entry (int);
void Probe_OpenMP_SetNumThreads_Exit (void);

void Probe_OpenMP_TaskID (long long id, int inst_or_exec);
void Probe_OpenMP_TaskLoopID (long long id);
void Probe_OpenMP_Task_Entry (UINT64 uf);
void Probe_OpenMP_Task_Exit (void);
void Probe_OpenMP_TaskUF_Entry (UINT64 uf);
void Probe_OpenMP_TaskUF_Exit (void);
void Probe_OpenMP_Taskwait_Entry (void);
void Probe_OpenMP_Taskwait_Exit (void);
void Probe_OpenMP_Taskgroup_start_Entry (void);
void Probe_OpenMP_Taskgroup_start_Exit (void);
void Probe_OpenMP_Taskgroup_end_Entry (void);
void Probe_OpenMP_Taskgroup_end_Exit (void);
void Probe_OpenMP_TaskLoop_Entry (void);
void Probe_OpenMP_TaskLoop_Exit (void);

void Probe_OpenMP_Ordered_Wait_Entry (void);
void Probe_OpenMP_Ordered_Wait_Exit (void);
void Probe_OpenMP_Ordered_Post_Entry (void);
void Probe_OpenMP_Ordered_Post_Exit (void);

/* OMPT probes */

void Probe_OMPT_Critical_Entry (void);
void Probe_OMPT_Critical_Exit (void);
void Probe_OMPT_Atomic_Entry (void);
void Probe_OMPT_Atomic_Exit (void);
void Probe_OMPT_Loop_Entry (void);
void Probe_OMPT_Loop_Exit (void);
void Probe_OMPT_Workshare_Entry (void);
void Probe_OMPT_Workshare_Exit (void);
void Probe_OMPT_Sections_Entry (void);
void Probe_OMPT_Sections_Exit (void);
void Probe_OMPT_Single_Entry (void);
void Probe_OMPT_Single_Exit (void);
void Probe_OMPT_Master_Entry (void);
void Probe_OMPT_Master_Exit (void);
void Probe_OMPT_Taskgroup_Entry (void);
void Probe_OMPT_Taskgroup_Exit (void);
void Probe_OMPT_OpenMP_TaskUF_Entry (UINT64 uf, UINT64 taskid);
void Probe_OMPT_OpenMP_TaskUF_Exit (UINT64 taskid);
void Probe_OMPT_dependence (uint64_t pred_task_id, uint64_t succ_task_id);

void Probe_OpenMP_Emit_numInstantiatedTasks (unsigned n);
void Probe_OpenMP_Emit_numExecutedTasks (unsigned n);

#endif
