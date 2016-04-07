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

#include "omp_probe.h"
#include "omp-common.h"
#include "wrapper.h"

#if !defined(HAVE__SYNC_ADD_AND_FETCH)
# ifdef HAVE_PTHREAD_H
#  include <pthread.h>
# endif
#endif

void Extrae_OpenMP_Join_NoWait_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Join_NoWait_Entry ();
}

void Extrae_OpenMP_Join_NoWait_Exit (void)
{
	Probe_OpenMP_Join_NoWait_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Join_Wait_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Join_Wait_Entry ();
}

void Extrae_OpenMP_Join_Wait_Exit (void)
{
	Probe_OpenMP_Join_Wait_Exit ();	
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_UF_Entry (const void *uf)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_UF_Entry ((UINT64) uf);
}

void Extrae_OpenMP_UF_Exit (void)
{
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Work_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Work_Entry ();
}

void Extrae_OpenMP_Work_Exit (void)
{
	Probe_OpenMP_Work_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_DO_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_DO_Entry ();
}

void Extrae_OpenMP_DO_Exit (void)
{
	Probe_OpenMP_DO_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Sections_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Sections_Entry ();
}

void Extrae_OpenMP_Sections_Exit (void)
{
	Probe_OpenMP_Sections_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_ParRegion_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_ParRegion_Entry ();
}

void Extrae_OpenMP_ParRegion_Exit (void)
{
	Probe_OpenMP_ParRegion_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_ParDO_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_ParDO_Entry ();
}

void Extrae_OpenMP_ParDO_Exit (void)
{
	Probe_OpenMP_ParDO_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_ParSections_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_ParSections_Entry ();
}

void Extrae_OpenMP_ParSections_Exit (void)
{
	Probe_OpenMP_ParSections_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Barrier_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Barrier_Entry ();
}

void Extrae_OpenMP_Barrier_Exit (void)
{
	Probe_OpenMP_Barrier_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Single_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Single_Entry ();
}

void Extrae_OpenMP_Single_Exit (void)
{
	Probe_OpenMP_Single_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Section_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Section_Entry ();
}

void Extrae_OpenMP_Section_Exit (void)
{
	Probe_OpenMP_Sections_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Named_Lock_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Named_Lock_Entry ();
}

void Extrae_OpenMP_Named_Lock_Exit (const void *name)
{
	Probe_OpenMP_Named_Lock_Exit (name);
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Named_Unlock_Entry (const void *name)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Named_Unlock_Entry (name);
}

void Extrae_OpenMP_Named_Unlock_Exit (void)
{
	Probe_OpenMP_Named_Unlock_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Unnamed_Lock_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Unnamed_Lock_Entry ();
}

void Extrae_OpenMP_Unnamed_Lock_Exit (void)
{
	Probe_OpenMP_Unnamed_Lock_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Unnamed_Unlock_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Unnamed_Unlock_Entry ();
}

void Extrae_OpenMP_Unnamed_Unlock_Exit (void)
{
	Probe_OpenMP_Unnamed_Unlock_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_GetNumThreads_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_GetNumThreads_Entry ();
}

void Extrae_OpenMP_GetNumThreads_Exit (void)
{
	Probe_OpenMP_GetNumThreads_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_SetNumThreads_Entry (int p1)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_SetNumThreads_Entry (p1);
}

void Extrae_OpenMP_SetNumThreads_Exit (void)
{
	Probe_OpenMP_SetNumThreads_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_TaskID (long long id)
{
	Probe_OpenMP_TaskID (id);
}

void Extrae_OpenMP_Task_Entry (const void* uf)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Task_Entry ((UINT64) uf);
}

void Extrae_OpenMP_Task_Exit (void)
{
	Probe_OpenMP_Task_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_TaskUF_Entry (const void* uf)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_TaskUF_Entry ((UINT64) uf);
}

void Extrae_OpenMP_TaskUF_Exit (void)
{
	Probe_OpenMP_TaskUF_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Taskwait_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Taskwait_Entry ();
}

void Extrae_OpenMP_Taskwait_Exit (void)
{
	Probe_OpenMP_Taskwait_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Taskgroup_start_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Taskgroup_start_Entry ();
}

void Extrae_OpenMP_Taskgroup_start_Exit (void)
{
	Probe_OpenMP_Taskgroup_start_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Taskgroup_end_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Taskgroup_end_Entry ();
}

void Extrae_OpenMP_Taskgroup_end_Exit (void)
{
	Probe_OpenMP_Taskgroup_end_Exit ();
	Backend_Leave_Instrumentation ();
}

/*
	OMPT added probes for OMPT events that do not match the previous events
*/

void Extrae_OMPT_Critical_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Critical_Entry ();
}

void Extrae_OMPT_Critical_Exit (void)
{
	Probe_OMPT_Critical_Exit();
	Backend_Leave_Instrumentation ();
}

void Extrae_OMPT_Atomic_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Atomic_Entry();	
}

void Extrae_OMPT_Atomic_Exit (void)
{
	Probe_OMPT_Atomic_Exit();
	Backend_Leave_Instrumentation ();
}

void Extrae_OMPT_Loop_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Loop_Entry();
}

void Extrae_OMPT_Loop_Exit (void)
{
	Probe_OMPT_Loop_Exit();
	Backend_Leave_Instrumentation ();
}

void Extrae_OMPT_Workshare_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Workshare_Entry ();
}

void Extrae_OMPT_Workshare_Exit (void)
{
	Probe_OMPT_Workshare_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OMPT_Sections_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Sections_Entry ();
}

void Extrae_OMPT_Sections_Exit (void)
{
	Probe_OMPT_Sections_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OMPT_Single_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Single_Entry ();
}

void Extrae_OMPT_Single_Exit (void)
{
	Probe_OMPT_Single_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OMPT_Master_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Master_Entry ();
}

void Extrae_OMPT_Master_Exit (void)
{
	Probe_OMPT_Master_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OMPT_Taskgroup_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_Taskgroup_Entry ();
}

void Extrae_OMPT_Taskgroup_Exit (void)
{
	Probe_OMPT_Taskgroup_Exit();
	Backend_Leave_Instrumentation();
}

void Extrae_OMPT_OpenMP_TaskUF_Entry (UINT64 uf, UINT64 taskid)
{
	Backend_Enter_Instrumentation (2);
	Probe_OMPT_OpenMP_TaskUF_Entry (uf, taskid);
}

void Extrae_OMPT_OpenMP_TaskUF_Exit (UINT64 taskid)
{
	Probe_OMPT_OpenMP_TaskUF_Exit (taskid);
	Backend_Leave_Instrumentation();
}

void Extrae_OMPT_dependence (uint64_t pred_task_id, uint64_t succ_task_id)
{
	Probe_OMPT_dependence (pred_task_id, succ_task_id);
}

static volatile unsigned Extrae_OpenMP_numInstantiatedTasks = 0;
static volatile unsigned Extrae_OpenMP_numExecutedTasks = 0;
#if !defined(HAVE__SYNC_FETCH_AND_ADD)
static pthread_mutex_t Extrae_OpenMP_numInstantiatedTasks_mtx =
	PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t Extrae_OpenMP_numExecutedTasks_mtx =
	PTHREAD_MUTEX_INITIALIZER;
#endif

void Extrae_OpenMP_Notify_NewInstantiatedTask (void)
{
#if defined(HAVE__SYNC_FETCH_AND_ADD)
	__sync_fetch_and_add (&Extrae_OpenMP_numInstantiatedTasks, 1);
#else
	pthread_mutex_lock (&Extrae_OpenMP_numInstantiatedTasks_mtx);
	Extrae_OpenMP_numInstantiatedTasks++;
	pthread_mutex_unlock (&Extrae_OpenMP_numInstantiatedTasks_mtx);
#endif
}

void Extrae_OpenMP_Notify_NewExecutedTask (void)
{
#if defined(HAVE__SYNC_FETCH_AND_ADD)
	__sync_fetch_and_add (&Extrae_OpenMP_numExecutedTasks, 1);
#else
	pthread_mutex_lock (&Extrae_OpenMP_numExecutedTasks_mtx);
	Extrae_OpenMP_numExecutedTasks++;
	pthread_mutex_unlock (&Extrae_OpenMP_numExecutedTasks_mtx);
#endif
}

void Extrae_OpenMP_EmitTaskStatistics (void)
{
	Probe_OpenMP_Emit_numInstantiatedTasks (Extrae_OpenMP_numInstantiatedTasks);
	Probe_OpenMP_Emit_numExecutedTasks (Extrae_OpenMP_numExecutedTasks);
}

