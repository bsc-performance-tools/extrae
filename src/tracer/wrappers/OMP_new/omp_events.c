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
#include "omp_common.h"
#include "omp_events.h"
#include "trace_macros_omp.h"

void Extrae_OpenMP_Counters (void)
{
	TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, HWC_EV, 0, xtr_OMP_check_config(OMP_COUNTERS_ENABLED));
}

void Extrae_OpenMP_Call_Entry (unsigned call_id)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_CALL_EV, EVT_BEGIN, call_id);
}

void Extrae_OpenMP_Call_Exit (unsigned call_id)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_CALL_EV, EVT_END, call_id);
}

void Extrae_OpenMP_Parallel_Entry (unsigned par_construct)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_PARALLEL_EV, par_construct, EMPTY);
}

void Extrae_OpenMP_Parallel_Exit ()
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_PARALLEL_EV, NEW_OMP_PARALLEL_END_VAL, EMPTY);
}

/* These forking routines used to emmit their own type but now they use the parallel type NEW_OMP_PARALLEL_EV */
void Extrae_OpenMP_Forking_Entry (unsigned par_construct)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_PARALLEL_EV, par_construct, EMPTY);
}

void Extrae_OpenMP_Forking_Exit ()
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_PARALLEL_EV, NEW_OMP_FORK_END_VAL, EMPTY);
}

void Extrae_OpenMP_Chunk_Entry (void)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_WSH_EV, NEW_OMP_WSH_NEXT_CHUNK_VAL, EMPTY);
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Chunk_Exit (void)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_WSH_EV, NEW_OMP_WSH_NEXT_CHUNK_END_VAL, EMPTY);
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Worksharing_Entry (unsigned wsh_construct)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_WSH_EV, wsh_construct, EMPTY);
}

void Extrae_OpenMP_Worksharing_Exit ()
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_WSH_EV, NEW_OMP_WSH_END_VAL, EMPTY);
}

void Extrae_OpenMP_Sync_Entry (unsigned sync_construct)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_SYNC_EV, sync_construct, EMPTY);
}

void Extrae_OpenMP_Sync_Exit (void)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_SYNC_EV, EVT_END, EMPTY);
}

void Extrae_OpenMP_Lock_Status (const void *name, unsigned lock_state)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_LOCK_EV, lock_state, EMPTY);
	if (name != NULL)
	{
		TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_LOCK_NAME_EV, (UINT64)name, EMPTY);
	}
	if ((xtr_OMP_check_config(OMP_ANNOTATE_CPU)) &&
			 ((lock_state == NEW_OMP_LOCK_TAKEN_VAL) ||
				(lock_state == NEW_OMP_LOCK_RELEASE_REQUEST_VAL)))
	{
		Extrae_AnnotateCPU ();
	}
}

void Extrae_OpenMP_Ordered (unsigned int ordered_state)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_ORDERED_EV, ordered_state, EMPTY);

	if ((xtr_OMP_check_config(OMP_ANNOTATE_CPU)) &&
			((ordered_state == NEW_OMP_ORDERED_WAIT_OVER_VAL) || 
			 (ordered_state == NEW_OMP_ORDERED_POST_START_VAL)))
	{
		Extrae_AnnotateCPU ();
	}
}

void Extrae_OpenMP_Taskgroup (unsigned int taskgroup_state)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASKGROUP_EV, taskgroup_state, EMPTY);
}

void Extrae_OpenMP_Outlined_Entry (const void *outlined_fn)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_NESTED_EV, omp_get_level(), EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_OUTLINED_ADDRESS_EV, (UINT64)outlined_fn, EMPTY);
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Outlined_Exit (void)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_NESTED_EV, omp_get_level(), EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_OUTLINED_ADDRESS_EV, EVT_END, EMPTY);
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Outlined_Entry_At (UINT64 time, const void *outlined_fn)
{
	TRACE_OMPEVENT(time, NEW_OMP_ADDRESS_EV, (UINT64)outlined_fn, EMPTY);
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Outlined_Exit_At ( UINT64 time )
{
  TRACE_OMPEVENT(time, NEW_OMP_ADDRESS_EV, EVT_END, EMPTY);
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Task_Inst_Entry (const void *task, long long task_id)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASKING_EV, NEW_OMP_TASK_INST_VAL, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_INST_ADDRESS_EV, (UINT64)task, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_INST_ID_EV, task_id, EMPTY);
	if ( xtr_OMP_check_config(OMP_TASK_DEPENDENCY_LINE_ENABLED) )
	{
		THREAD_TRACE_USER_COMMUNICATION_EVENT(THREADID, LAST_READ_TIME, USER_SEND_EV, TASKID, 0, task_id, task_id);
	}
}

void Extrae_OpenMP_Task_Inst_Exit (void)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASKING_EV, EVT_END, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_INST_ADDRESS_EV, EVT_END, EMPTY);
}

void Extrae_OpenMP_Task_Exec_Entry (int task_or_taskloop, const void *task, long long task_id)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASKING_EV, task_or_taskloop, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_EXEC_ADDRESS_EV, (UINT64)task, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_EXEC_ID_EV, task_id, EMPTY);
	if ( xtr_OMP_check_config(task_or_taskloop == NEW_OMP_TASK_EXEC_VAL? OMP_TASK_DEPENDENCY_LINE_ENABLED : OMP_TASKLOOP_DEPENDENCY_LINE_ENABLED) )
	{
		THREAD_TRACE_USER_COMMUNICATION_EVENT(THREADID, LAST_READ_TIME, USER_RECV_EV, TASKID, 0, task_id, task_id);
	}
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Task_Exec_Exit (void)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASKING_EV, EVT_END, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_EXEC_ADDRESS_EV, EVT_END, EMPTY);
	if (xtr_OMP_check_config(OMP_ANNOTATE_CPU))
	{
		Extrae_AnnotateCPU (); 
	}
}

void Extrae_OpenMP_Taskloop_Entry (const void *task, long long taskloop_id, int num_tasks)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASKING_EV, NEW_OMP_TASKLOOP_INST_VAL, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_INST_ADDRESS_EV, (UINT64)task, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_INST_ID_EV, taskloop_id, EMPTY);
	if ( xtr_OMP_check_config(OMP_TASK_DEPENDENCY_LINE_ENABLED) )
	{
		THREAD_TRACE_SAME_N_USER_COMMUNICATION_EVENT(THREADID, LAST_READ_TIME, num_tasks, USER_SEND_EV, TASKID, 0, taskloop_id, taskloop_id);
	}
}

void Extrae_OpenMP_Taskloop_Exit (void)
{
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASKING_EV, EVT_END, EMPTY);
	TRACE_OMPEVENT(LAST_READ_TIME, NEW_OMP_TASK_INST_ADDRESS_EV, EVT_END, EMPTY);
}

