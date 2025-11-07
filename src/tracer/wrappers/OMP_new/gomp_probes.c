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
#include "gomp_helpers.h"
#include "gomp_probes.h"
#include "trace_macros.h"
#include "omp_stats.h"

/**
 * Outlined routines
 */

void xtr_probe_entry_GOMP_parallel_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_REGION_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_start_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_REGION_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_start_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_static_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_STATIC_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_static_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_static_start_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_STATIC_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_static_start_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_dynamic_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_DYNAMIC_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_dynamic_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_dynamic_start_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_DYNAMIC_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_dynamic_start_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_dynamic_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_DYNAMIC_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_dynamic_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_guided_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_GUIDED_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_guided_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_guided_start_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_GUIDED_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_guided_start_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_guided_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_GUIDED_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_guided_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_runtime_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_runtime_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_runtime_start_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_runtime_start_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_maybe_nonmonotonic_runtime_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_maybe_nonmonotonic_runtime_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_runtime_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_runtime_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_task_OL(struct task_helper_t *task_helper)
{
	Extrae_OpenMP_Task_Exec_Entry(NEW_OMP_TASK_EXEC_VAL, task_helper->fn, task_helper->id);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_task_OL()
{
	Extrae_OpenMP_Task_Exec_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_taskloop_OL(struct taskloop_helper_t *taskloop_helper)
{
	Extrae_OpenMP_Task_Exec_Entry(NEW_OMP_TASKLOOP_EXEC_VAL, taskloop_helper->fn, taskloop_helper->id);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_taskloop_OL()
{
	Extrae_OpenMP_Task_Exec_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_taskloop_ull_OL(struct taskloop_helper_t *taskloop_helper)
{
	Extrae_OpenMP_Task_Exec_Entry(NEW_OMP_TASKLOOP_EXEC_VAL, taskloop_helper->fn, taskloop_helper->id);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_taskloop_ull_OL()
{
	Extrae_OpenMP_Task_Exec_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_sections_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_SECTIONS_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_SECTION_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_sections_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_parallel_sections_start_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_SECTIONS_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_SECTION_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_parallel_sections_start_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

void xtr_probe_entry_GOMP_teams_reg_OL(struct parallel_helper_t *par_helper)
{
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_TEAMS_VAL);
	Extrae_OpenMP_Outlined_Entry(par_helper->fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_exit_GOMP_teams_reg_OL()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

/**
 * GOMP barrier
 */

void xtr_probe_entry_GOMP_barrier()
{
	Extrae_OpenMP_Call_Entry(GOMP_BARRIER_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_BARRIER_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_barrier()
{
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_BARRIER_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

/**
 * GOMP critical
 * GOMP atomic
 */

void xtr_probe_entry_GOMP_critical_start()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Call_Entry(GOMP_CRITICAL_START_VAL);
		Extrae_OpenMP_Sync_Entry(NEW_OMP_LOCK_CRITICAL_VAL);
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}

void xtr_probe_exit_GOMP_critical_start()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_TAKEN_VAL);
		Extrae_OpenMP_Call_Exit(GOMP_CRITICAL_START_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}

void xtr_probe_entry_GOMP_critical_end()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Call_Entry(GOMP_CRITICAL_END_VAL);
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_RELEASE_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}

void xtr_probe_exit_GOMP_critical_end()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_RELEASED_VAL);
		Extrae_OpenMP_Sync_Exit();
		Extrae_OpenMP_Call_Exit(GOMP_CRITICAL_END_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}

void xtr_probe_entry_GOMP_critical_name_start(void **pptr)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Call_Entry(GOMP_CRITICAL_NAME_START_VAL);
		Extrae_OpenMP_Sync_Entry(NEW_OMP_LOCK_CRITICAL_NAMED_VAL);
		Extrae_OpenMP_Lock_Status(pptr, NEW_OMP_LOCK_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}

void xtr_probe_exit_GOMP_critical_name_start(void **pptr)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Lock_Status(pptr, NEW_OMP_LOCK_TAKEN_VAL);
		Extrae_OpenMP_Call_Exit(GOMP_CRITICAL_NAME_START_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}

void xtr_probe_entry_GOMP_critical_name_end(void **pptr)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Call_Entry(GOMP_CRITICAL_NAME_END_VAL);
		Extrae_OpenMP_Lock_Status(pptr, NEW_OMP_LOCK_RELEASE_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}

void xtr_probe_exit_GOMP_critical_name_end(void **pptr)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Lock_Status(pptr, NEW_OMP_LOCK_RELEASED_VAL);
		Extrae_OpenMP_Sync_Exit();
		Extrae_OpenMP_Call_Exit(GOMP_CRITICAL_NAME_END_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}

void xtr_probe_entry_GOMP_atomic_start()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Call_Entry(GOMP_ATOMIC_START_VAL);
		Extrae_OpenMP_Sync_Entry(NEW_OMP_LOCK_ATOMIC_VAL);
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}

void xtr_probe_exit_GOMP_atomic_start()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_TAKEN_VAL);
		Extrae_OpenMP_Call_Exit(GOMP_ATOMIC_START_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}

void xtr_probe_entry_GOMP_atomic_end()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Call_Entry(GOMP_ATOMIC_END_VAL);
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_RELEASE_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}

void xtr_probe_exit_GOMP_atomic_end()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) ) 
	{
		Extrae_OpenMP_Lock_Status(NULL, NEW_OMP_LOCK_RELEASED_VAL);
		Extrae_OpenMP_Sync_Exit();
		Extrae_OpenMP_Call_Exit(GOMP_ATOMIC_END_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}

/**
 * GOMP loop routines
 */

void xtr_probe_entry_GOMP_loop_static_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_STATIC_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_STATIC_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_static_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_STATIC_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_dynamic_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_DYNAMIC_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_DYNAMIC_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_dynamic_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_DYNAMIC_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_guided_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_GUIDED_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_GUIDED_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_guided_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_GUIDED_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_runtime_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_RUNTIME_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_runtime_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_RUNTIME_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_nonmonotonic_dynamic_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_NONMONOTONIC_DYNAMIC_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_DYNAMIC_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_nonmonotonic_dynamic_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_NONMONOTONIC_DYNAMIC_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_nonmonotonic_guided_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_NONMONOTONIC_GUIDED_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_GUIDED_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_nonmonotonic_guided_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_NONMONOTONIC_GUIDED_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_nonmonotonic_runtime_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_NONMONOTONIC_RUNTIME_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_nonmonotonic_runtime_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_NONMONOTONIC_RUNTIME_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_maybe_nonmonotonic_runtime_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_maybe_nonmonotonic_runtime_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_static_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_STATIC_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_ORDERED_STATIC_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_static_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_STATIC_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_dynamic_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_DYNAMIC_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_ORDERED_DYNAMIC_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_dynamic_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_DYNAMIC_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_guided_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_GUIDED_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_ORDERED_GUIDED_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_guided_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_GUIDED_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_runtime_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_RUNTIME_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_ORDERED_RUNTIME_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_runtime_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_RUNTIME_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_static_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_STATIC_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_static_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_STATIC_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_dynamic_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_DYNAMIC_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_dynamic_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_DYNAMIC_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_guided_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_GUIDED_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_guided_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_GUIDED_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_runtime_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_RUNTIME_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_runtime_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_RUNTIME_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_nonmonotonic_dynamic_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_NONMONOTONIC_DYNAMIC_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_nonmonotonic_dynamic_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_NONMONOTONIC_DYNAMIC_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_nonmonotonic_guided_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_NONMONOTONIC_GUIDED_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_nonmonotonic_guided_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_NONMONOTONIC_GUIDED_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_nonmonotonic_runtime_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_NONMONOTONIC_RUNTIME_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_nonmonotonic_runtime_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_NONMONOTONIC_RUNTIME_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_maybe_nonmonotonic_runtime_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_maybe_nonmonotonic_runtime_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_MAYBE_NONMONOTONIC_RUNTIME_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_static_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_STATIC_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_static_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_STATIC_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_dynamic_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_DYNAMIC_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_dynamic_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_DYNAMIC_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_guided_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_GUIDED_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_guided_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_GUIDED_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_ordered_runtime_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_ORDERED_RUNTIME_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_ordered_runtime_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_ORDERED_RUNTIME_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_doacross_static_start(unsigned ncounts)
{
	__GOMP_save_doacross_ncounts(ncounts);
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_DOACROSS_STATIC_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DOACROSS_STATIC_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_doacross_static_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_DOACROSS_STATIC_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_doacross_dynamic_start(unsigned ncounts)
{
	__GOMP_save_doacross_ncounts(ncounts);
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_DOACROSS_DYNAMIC_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DOACROSS_DYNAMIC_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_doacross_dynamic_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_DOACROSS_DYNAMIC_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_doacross_guided_start(unsigned ncounts)
{
	__GOMP_save_doacross_ncounts(ncounts);
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_DOACROSS_GUIDED_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DOACROSS_GUIDED_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_doacross_guided_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_DOACROSS_GUIDED_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_doacross_runtime_start(unsigned ncounts)
{
	__GOMP_save_doacross_ncounts(ncounts);
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_DOACROSS_RUNTIME_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DOACROSS_RUNTIME_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_doacross_runtime_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_DOACROSS_RUNTIME_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_static_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_STATIC_START_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_static_start(const void *outlined_fn)
{
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_STATIC_START_VAL);
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_STATIC_VAL);
	Extrae_OpenMP_Outlined_Entry(outlined_fn);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_dynamic_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_DYNAMIC_START_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_dynamic_start(const void *outlined_fn)
{
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_DYNAMIC_START_VAL);
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_DYNAMIC_VAL);
	Extrae_OpenMP_Outlined_Entry(outlined_fn);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_guided_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_GUIDED_START_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_guided_start(const void *outlined_fn)
{
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_GUIDED_START_VAL);
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_GUIDED_VAL);
	Extrae_OpenMP_Outlined_Entry(outlined_fn);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_runtime_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_RUNTIME_START_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_runtime_start(const void *outlined_fn)
{
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_RUNTIME_START_VAL);
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_LOOP_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_DO_RUNTIME_VAL);
	Extrae_OpenMP_Outlined_Entry(outlined_fn);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_static()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_STATIC_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_static()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_STATIC_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_dynamic()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_DYNAMIC_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_dynamic()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_DYNAMIC_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_guided()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_GUIDED_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_guided()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_GUIDED_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_runtime()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_RUNTIME_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_runtime()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_RUNTIME_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_dynamic()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_NONMONOTONIC_DYNAMIC_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_dynamic()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_NONMONOTONIC_DYNAMIC_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_guided()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_NONMONOTONIC_GUIDED_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_guided()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_NONMONOTONIC_GUIDED_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_runtime()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_NONMONOTONIC_RUNTIME_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_runtime()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_NONMONOTONIC_RUNTIME_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_loop_maybe_nonmonotonic_runtime()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_LOOP_MAYBE_NONMONOTONIC_RUNTIME_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_LOOP_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_loop_maybe_nonmonotonic_runtime()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_LOOP_MAYBE_NONMONOTONIC_RUNTIME_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_loop_end()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_END_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_JOIN_WAIT_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_loop_end()
{
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Worksharing_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_END_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

void xtr_probe_entry_GOMP_loop_end_nowait()
{
	Extrae_OpenMP_Call_Entry(GOMP_LOOP_END_NOWAIT_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_JOIN_NOWAIT_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_loop_end_nowait() 
{
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Worksharing_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_LOOP_END_NOWAIT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/**
 * GOMP ordered
 */

void xtr_probe_entry_GOMP_ordered_start()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Call_Entry(GOMP_ORDERED_START_VAL);
		Extrae_OpenMP_Sync_Entry(NEW_OMP_ORDERED_VAL);
		Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_WAIT_START_VAL);
		xtr_stats_OMP_update_overhead_entry();
	}
}

void xtr_probe_exit_GOMP_ordered_start()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_WAIT_OVER_VAL);
		Extrae_OpenMP_Call_Exit(GOMP_ORDERED_START_VAL);
		xtr_stats_OMP_update_overhead_exit();
	}
}

void xtr_probe_entry_GOMP_ordered_end()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Call_Entry(GOMP_ORDERED_END_VAL);
		Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_POST_START_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}

void xtr_probe_exit_GOMP_ordered_end()
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_POST_READY_VAL);
		Extrae_OpenMP_Sync_Exit();
		Extrae_OpenMP_Call_Exit(GOMP_ORDERED_END_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}

/**
 * GOMP doacross
 */

void xtr_probe_entry_GOMP_doacross_post()
{
	Extrae_OpenMP_Call_Entry(GOMP_DOACROSS_POST_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_POST_VAL);
	Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_POST_START_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_doacross_post()
{
	Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_POST_READY_VAL);
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_DOACROSS_POST_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

void xtr_probe_entry_GOMP_doacross_wait()
{
	Extrae_OpenMP_Call_Entry(GOMP_DOACROSS_WAIT_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_WAIT_VAL);
	Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_WAIT_START_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_doacross_wait()
{
	Extrae_OpenMP_Ordered(NEW_OMP_ORDERED_WAIT_OVER_VAL);
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_DOACROSS_WAIT_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

/**
 * GOMP parallel
 */

void xtr_probe_entry_GOMP_parallel_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_START_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_REGION_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}


void xtr_probe_exit_GOMP_parallel_start(const void *outlined_fn)
{
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_START_VAL);
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_REGION_VAL);
	
	/* 
	 * OpenMP runtime does not invoke the callback for the master thread, 
	 * the compiler transformation adds an explicit call to the 'outlined_fn', 
	 * so the master thread must emit this event directly 
	 */
	Extrae_OpenMP_Outlined_Entry(outlined_fn);
	xtr_stats_OMP_update_par_OL_entry();
}

/* GCC4 ONLY */
void xtr_probe_entry_GOMP_parallel_end()
{
	Extrae_OpenMP_Outlined_Exit();
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_END_VAL);
	Extrae_OpenMP_Parallel_Exit();
	xtr_stats_OMP_update_par_OL_exit();
}

/* GCC4 ONLY */
void xtr_probe_exit_GOMP_parallel_end()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_END_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_REGION_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/**
 * GOMP task
 */

void xtr_probe_entry_GOMP_task(struct task_helper_t *task_helper)
{
	Extrae_OpenMP_Call_Entry(GOMP_TASK_VAL);
	Extrae_OpenMP_Task_Inst_Entry(task_helper->fn, task_helper->id);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_task()
{
	Extrae_OpenMP_Task_Inst_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TASK_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/**
 * GOMP taskloop
 */

void xtr_probe_entry_GOMP_taskloop(struct taskloop_helper_t *taskloop_helper, int num_tasks)
{
	Extrae_OpenMP_Call_Entry(GOMP_TASKLOOP_VAL);
	Extrae_OpenMP_Taskloop_Entry(taskloop_helper->fn, taskloop_helper->id, num_tasks);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_taskloop()
{
	Extrae_OpenMP_Taskloop_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TASKLOOP_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_taskloop_ull(struct taskloop_helper_t *taskloop_helper, int num_tasks)
{
	Extrae_OpenMP_Call_Entry(GOMP_TASKLOOP_VAL);
	Extrae_OpenMP_Taskloop_Entry(taskloop_helper->fn, taskloop_helper->id, num_tasks);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_taskloop_ull()
{
	Extrae_OpenMP_Taskloop_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TASKLOOP_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/**
 * GOMP taskwait
 */

void xtr_probe_entry_GOMP_taskwait()
{
	Extrae_OpenMP_Call_Entry(GOMP_TASKWAIT_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_TASKWAIT_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_taskwait()
{
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TASKWAIT_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

/**
 * GOMP taskyield
 */
void xtr_probe_entry_GOMP_taskyield()
{
	Extrae_OpenMP_Call_Entry(GOMP_TASKYIELD_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_TASKYIELD_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_taskyield()
{
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TASKYIELD_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

/**
 * GOMP taskgroup
 */

void xtr_probe_entry_GOMP_taskgroup_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_TASKGROUP_START_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_TASKGROUP_VAL);
	Extrae_OpenMP_Taskgroup(NEW_OMP_TASKGROUP_OPENING_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_taskgroup_start()
{
	Extrae_OpenMP_Taskgroup(NEW_OMP_TASKGROUP_ENTERING_VAL);
	Extrae_OpenMP_Call_Exit(GOMP_TASKGROUP_START_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

void xtr_probe_entry_GOMP_taskgroup_end()
{
	Extrae_OpenMP_Call_Entry(GOMP_TASKGROUP_END_VAL);
	Extrae_OpenMP_Taskgroup(NEW_OMP_TASKGROUP_WAITING_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_taskgroup_end()
{
	Extrae_OpenMP_Taskgroup(NEW_OMP_TASKGROUP_END_VAL);
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TASKGROUP_END_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

/**
 * GOMP sections
 */

void xtr_probe_entry_GOMP_sections_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_SECTIONS_START_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_SECTION_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_sections_start()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_SECTIONS_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_sections_next()
{
	Extrae_OpenMP_Call_Entry(GOMP_SECTIONS_NEXT_VAL);
	Extrae_OpenMP_Chunk_Entry();
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_sections_next()
{
	Extrae_OpenMP_Chunk_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_SECTIONS_NEXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_parallel_sections_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_SECTIONS_START_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_SECTIONS_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_sections_start(const void *outlined_fn)
{
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_SECTIONS_START_VAL);
	Extrae_OpenMP_Parallel_Entry(NEW_OMP_PARALLEL_SECTIONS_VAL);
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_SECTION_VAL);
	/*
	 * OpenMP runtime does not invoke the callback for the master thread, 
	 * the compiler transformation adds an explicit call to the 'outlined_fn', 
	 * so the master thread must emit this event directly 
	 */
	Extrae_OpenMP_Outlined_Entry(outlined_fn);
	xtr_stats_OMP_update_par_OL_entry();
}

void xtr_probe_entry_GOMP_parallel_sections()
{
	Extrae_OpenMP_Call_Entry(GOMP_PARALLEL_SECTIONS_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_PARALLEL_SECTIONS_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_parallel_sections()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_PARALLEL_SECTIONS_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_sections_end()
{
	Extrae_OpenMP_Call_Entry(GOMP_SECTIONS_END_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_JOIN_WAIT_VAL);
	xtr_stats_OMP_update_synchronization_entry();
}

void xtr_probe_exit_GOMP_sections_end()
{
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Worksharing_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_SECTIONS_END_VAL);
	xtr_stats_OMP_update_synchronization_exit();
}

void xtr_probe_entry_GOMP_sections_end_nowait()
{
	Extrae_OpenMP_Call_Entry(GOMP_SECTIONS_END_NOWAIT_VAL);
	Extrae_OpenMP_Sync_Entry(NEW_OMP_JOIN_NOWAIT_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_sections_end_nowait()
{
	Extrae_OpenMP_Sync_Exit();
	Extrae_OpenMP_Worksharing_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_SECTIONS_END_NOWAIT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/*
 * GOMP single
 */

void xtr_probe_entry_GOMP_single_start()
{
	Extrae_OpenMP_Call_Entry(GOMP_SINGLE_START_VAL);

	/*
	 * Other GOMP_*_start routines emit a CHUNK event, but we don't in single
	 * because the same exact interval is marked with WSH SINGLE, as there's no
	 * corresponding GOMP_*_end call.
	 */
	Extrae_OpenMP_Worksharing_Entry(NEW_OMP_WSH_SINGLE_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_single_start()
{
	Extrae_OpenMP_Worksharing_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_SINGLE_START_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/**
 * GOMP teams
 */

void xtr_probe_entry_GOMP_teams_reg()
{
	Extrae_OpenMP_Call_Entry(GOMP_TEAMS_REG_VAL);
	Extrae_OpenMP_Forking_Entry(NEW_OMP_TEAMS_FORK_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_teams_reg()
{
	Extrae_OpenMP_Forking_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TEAMS_REG_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/**
 * GOMP target
 */

void xtr_probe_entry_GOMP_target()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_target_data()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_DATA_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_DATA_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target_data()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_DATA_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_target_end_data()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_END_DATA_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_DATA_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target_end_data()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_END_DATA_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_target_update()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_UPDATE_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_UPDATE_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target_update()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_UPDATE_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_target_enter_exit_data()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_ENTER_EXIT_DATA_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_ENTER_DATA_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target_enter_exit_data()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_ENTER_EXIT_DATA_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_target_ext()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_EXT_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target_ext()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_EXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_target_data_ext()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_DATA_EXT_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_DATA_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target_data_ext()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_DATA_EXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

void xtr_probe_entry_GOMP_target_update_ext()
{
	Extrae_OpenMP_Call_Entry(GOMP_TARGET_UPDATE_EXT_VAL);
	Extrae_OpenMP_Target_Entry(NEW_OMP_TARGET_UPDATE_VAL);
	xtr_stats_OMP_update_overhead_entry();
}

void xtr_probe_exit_GOMP_target_update_ext()
{
	Extrae_OpenMP_Target_Exit();
	Extrae_OpenMP_Call_Exit(GOMP_TARGET_UPDATE_EXT_VAL);
	xtr_stats_OMP_update_overhead_exit();
}

/**
 * omp_set_num_threads
 */

void xtr_probe_exit_omp_set_num_threads(int num_threads)
{
	Backend_ChangeNumberOfThreads(num_threads);
}

void xtr_probe_exit_omp_set_num_threads_( int ) __attribute__ ((alias( "xtr_probe_exit_omp_set_num_threads" )));

/**
 * omp_set_lock
 * omp_unset_lock
 */

void xtr_probe_entry_omp_set_lock(void *lock)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Call_Entry(GOMP_SET_LOCK_VAL);
		Extrae_OpenMP_Sync_Entry(NEW_OMP_LOCK_CRITICAL_NAMED_VAL);
		Extrae_OpenMP_Lock_Status(lock, NEW_OMP_LOCK_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}
void xtr_probe_entry_omp_set_lock_( void * ) __attribute__ ((alias( "xtr_probe_entry_omp_set_lock" )));

void xtr_probe_exit_omp_set_lock(void *lock)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Lock_Status(lock, NEW_OMP_LOCK_TAKEN_VAL);
		Extrae_OpenMP_Call_Exit(GOMP_SET_LOCK_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}
void xtr_probe_exit_omp_set_lock_( void * ) __attribute__ ((alias( "xtr_probe_exit_omp_set_lock" )));

void xtr_probe_entry_omp_unset_lock(void *lock)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Call_Entry(GOMP_UNSET_LOCK_VAL);
		Extrae_OpenMP_Lock_Status(lock, NEW_OMP_LOCK_RELEASE_REQUEST_VAL);
		xtr_stats_OMP_update_synchronization_entry();
	}
}
void xtr_probe_entry_omp_unset_lock_( void * ) __attribute__ ((alias( "xtr_probe_entry_omp_unset_lock" )));

void xtr_probe_exit_omp_unset_lock(void *lock)
{
	if( xtr_OMP_check_config(OMP_LOCKS_ENABLED) )
	{
		Extrae_OpenMP_Lock_Status(lock, NEW_OMP_LOCK_RELEASED_VAL);
		Extrae_OpenMP_Sync_Exit();
		Extrae_OpenMP_Call_Exit(GOMP_UNSET_LOCK_VAL);
		xtr_stats_OMP_update_synchronization_exit();
	}
}
void xtr_probe_exit_omp_unset_lock_( void * ) __attribute__ ((alias( "xtr_probe_exit_omp_unset_lock" )));
