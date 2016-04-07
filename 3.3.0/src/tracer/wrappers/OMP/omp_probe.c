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

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "omp_probe.h"

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

static int TraceOMPLocks = FALSE;

void setTrace_OMPLocks (int value)
{
	TraceOMPLocks = value;
}

int getTrace_OMPLocks (void)
{
	return TraceOMPLocks;
}

void Probe_OpenMP_Join_NoWait_Entry (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, JOIN_EV, EVT_BEGIN, JOIN_NOWAIT_VAL);
}

void Probe_OpenMP_Join_NoWait_Exit (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, JOIN_EV, EVT_END, JOIN_NOWAIT_VAL);
}

void Probe_OpenMP_Join_Wait_Entry (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, JOIN_EV, EVT_BEGIN, JOIN_WAIT_VAL);
}

void Probe_OpenMP_Join_Wait_Exit (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, JOIN_EV, EVT_END, JOIN_WAIT_VAL);
}

void Probe_OpenMP_UF_Entry (UINT64 uf)
{
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPFUNC_EV, uf, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_UF_Exit (void)
{
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPFUNC_EV, EVT_END, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_Work_Entry (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, WORK_EV, EVT_BEGIN, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_Work_Exit (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, WORK_EV, EVT_END, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_DO_Entry (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, WSH_EV, WSH_DO_VAL, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_DO_Exit (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_END_VAL, EMPTY); 
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_Sections_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, WSH_EV, WSH_SEC_VAL, EMPTY);
}

void Probe_OpenMP_Sections_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_END_VAL, EMPTY); 
}

void Probe_OpenMP_ParRegion_Entry (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS (LAST_READ_TIME, PAR_EV, PAR_REG_VAL, EMPTY);
		Extrae_AnnotateCPU (LAST_READ_TIME);
	}
}

void Probe_OpenMP_ParRegion_Exit (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_END_VAL, EMPTY);
		Extrae_AnnotateCPU (LAST_READ_TIME);
	}
}

void Probe_OpenMP_ParDO_Entry (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, PAR_EV, PAR_WSH_VAL, EMPTY);
		Extrae_AnnotateCPU (LAST_READ_TIME);
	}
}

void Probe_OpenMP_ParDO_Exit (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_END_VAL, EMPTY);
		Extrae_AnnotateCPU (LAST_READ_TIME);
	}
}

void Probe_OpenMP_ParSections_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, PAR_EV, PAR_SEC_VAL, EMPTY);
}

void Probe_OpenMP_ParSections_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_END_VAL, EMPTY);
}

void Probe_OpenMP_Barrier_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, BARRIEROMP_EV, EVT_BEGIN, EMPTY);
}

void Probe_OpenMP_Barrier_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, BARRIEROMP_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_Single_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, WSH_EV, WSH_SINGLE_VAL, EMPTY);
}

void Probe_OpenMP_Single_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_Master_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, WSH_EV, WSH_MASTER_VAL, EMPTY);
}

void Probe_OpenMP_Master_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_Section_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, WSH_EV, WSH_SEC_VAL, EMPTY);
}

void Probe_OpenMP_Section_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_Named_Lock_Entry (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, NAMEDCRIT_EV, LOCK_VAL, EMPTY);
}

void Probe_OpenMP_Named_Lock_Exit (const void *name)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, NAMEDCRIT_EV, LOCKED_VAL, (UINT64) name);
}

void Probe_OpenMP_Named_Unlock_Entry (const void *name)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, NAMEDCRIT_EV, UNLOCK_VAL, (UINT64) name);
}

void Probe_OpenMP_Named_Unlock_Exit (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, NAMEDCRIT_EV, UNLOCKED_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Lock_Entry (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, UNNAMEDCRIT_EV, LOCK_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Lock_Exit (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, UNNAMEDCRIT_EV, LOCKED_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Unlock_Entry (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, UNNAMEDCRIT_EV, UNLOCK_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Unlock_Exit (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, UNNAMEDCRIT_EV, UNLOCKED_VAL, EMPTY);
}

void Probe_OpenMP_GetNumThreads_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPGETNUMTHREADS_EV, EVT_BEGIN, EMPTY); 
}

void Probe_OpenMP_GetNumThreads_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPGETNUMTHREADS_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_SetNumThreads_Entry (int p1)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPSETNUMTHREADS_EV, EVT_BEGIN, p1); 
}

void Probe_OpenMP_SetNumThreads_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPSETNUMTHREADS_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_TaskID (long long id)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, TASKID_EV, EVT_BEGIN, (UINT64) id);
	}
}

void Probe_OpenMP_Task_Entry (UINT64 uf)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, TASK_EV, uf, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_Task_Exit (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, TASK_EV, EVT_END, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_TaskUF_Entry (UINT64 uf)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, TASKFUNC_EV, uf, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_TaskUF_Exit (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, TASKFUNC_EV, EVT_END, EMPTY);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OpenMP_Taskwait_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, TASKWAIT_EV, EVT_BEGIN, EMPTY);
}

void Probe_OpenMP_Taskwait_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, TASKWAIT_EV, EVT_END, EMPTY);
}

void Probe_OpenMP_Taskgroup_start_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, TASKGROUP_START_EV, EVT_BEGIN, EMPTY);
}

void Probe_OpenMP_Taskgroup_start_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, TASKGROUP_START_EV, EVT_END, EMPTY);
}

void Probe_OpenMP_Taskgroup_end_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, TASKGROUP_END_EV, EVT_BEGIN, EMPTY);
}

void Probe_OpenMP_Taskgroup_end_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, TASKGROUP_END_EV, EVT_END, EMPTY);
}

/*
	OMPT added probes for OMPT events that do not match the previous events
*/


void Probe_OMPT_Critical_Entry (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_CRITICAL_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Critical_Exit (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_CRITICAL_EV, EVT_END, EMPTY);
}

void Probe_OMPT_Atomic_Entry (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_ATOMIC_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Atomic_Exit (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_ATOMIC_EV, EVT_END, EMPTY);
}

void Probe_OMPT_Loop_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_LOOP_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Loop_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_LOOP_EV, EVT_END, EMPTY);
}

void Probe_OMPT_Workshare_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_WORKSHARE_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Workshare_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_WORKSHARE_EV, EVT_END, EMPTY);
}

void Probe_OMPT_Sections_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_SECTIONS_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Sections_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_SECTIONS_EV, EVT_END, EMPTY);
}

void Probe_OMPT_Single_Entry (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_SINGLE_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Single_Exit (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_SINGLE_EV, EVT_END, EMPTY);
}

void Probe_OMPT_Master_Entry (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_MASTER_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Master_Exit (void)
{
	DEBUG
	if (mpitrace_on && TraceOMPLocks)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_MASTER_EV, EVT_END, EMPTY);
}

void Probe_OMPT_Taskgroup_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_TASKGROUP_IN_EV, EVT_BEGIN, EMPTY);
}

void Probe_OMPT_Taskgroup_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_TASKGROUP_IN_EV, EVT_END, EMPTY);
}

void Probe_OMPT_OpenMP_TaskUF_Entry (UINT64 uf, UINT64 taskid)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(LAST_READ_TIME, OMPT_TASKFUNC_EV, uf, taskid);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OMPT_OpenMP_TaskUF_Exit (UINT64 taskid)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPT_TASKFUNC_EV, EVT_END, taskid);
		/*Extrae_AnnotateCPU (LAST_READ_TIME);*/
	}
}

void Probe_OMPT_dependence (uint64_t pred_task_id, uint64_t succ_task_id)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENT2PARAM(TIME, OMPT_DEPENDENCE_EV, 0, pred_task_id,
		  succ_task_id);
}

void Probe_OpenMP_Emit_numInstantiatedTasks (unsigned n)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENT(LAST_READ_TIME, OMP_STATS_EV,
		  OMP_NUM_TASKS_INSTANTIATED, n);
}

void Probe_OpenMP_Emit_numExecutedTasks (unsigned n)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENT(LAST_READ_TIME, OMP_STATS_EV,
		  OMP_NUM_TASKS_EXECUTED, n);
}
