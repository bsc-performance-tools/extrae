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

#define _GNU_SOURCE
#include "common.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
# define DBG fprintf(stderr, "Captured %s\n", __func__);
#else
# define DBG
#endif

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "clock.h"
#include "events.h"
#include "taskid.h"
#include "wrapper.h"

#include "gpi_events.h"
#include "gpi_wrapper.h"

static gaspi_rank_t
Extrae_GPI_NumTasks()
{
	static int run = FALSE;
	static gaspi_rank_t mysize;

	if (!run)
	{
		pgaspi_proc_num(&mysize);
		run = TRUE;
	}

	return (gaspi_rank_t)mysize;
}

static gaspi_rank_t
Extrae_GPI_TaskID()
{
	static int run = FALSE;
	static gaspi_rank_t myrank;

	if (!run)
	{
		pgaspi_proc_rank(&myrank);
		run = TRUE;
	}

	return (gaspi_rank_t)myrank;
}

static void
Extrae_GPI_Barrier()
{
	// XXX Change GASPI_BLOCK to actual timeout
	pgaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);
}

static void
Extrae_GPI_Finalize()
{
	// XXX Change GASPI_BLOCK to actual timeout
	pgaspi_proc_term(GASPI_BLOCK);
}

/*
 * GPI_remove_file_list
 */
void
GPI_remove_file_list(int all)
{
	char tmpname[1024];

	if (all || (!all && TASKID == 0))
	{
		sprintf(tmpname, "%s/%s%s", final_dir, appl_name, EXT_MPITS);
		unlink(tmpname);
	}
}

/*
 * GASPI Wrappers
 */
gaspi_return_t
gaspi_proc_init(gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;
	iotimer_t GPI_Init_start_time, GPI_Init_end_time;

	ret = pgaspi_proc_init(timeout_ms);

	Extrae_set_ApplicationIsGPI(TRUE);

	/* Setup callbacks for TASK identification and barrier execution */
	Extrae_set_taskid_function((unsigned int (*)(void))Extrae_GPI_TaskID);
	Extrae_set_numtasks_function((unsigned int (*)(void))Extrae_GPI_NumTasks);
	Extrae_set_barrier_tasks_function(Extrae_GPI_Barrier);
	Extrae_set_finalize_task_function(Extrae_GPI_Finalize);

	if (Extrae_is_initialized_Wrapper() != EXTRAE_NOT_INITIALIZED)
	{
		Backend_updateTaskID();
	}

	/*
	 * Generate a tentative file list, remove first if the list was generated
	 * by Extrae_init
	 */
	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_EXTRAE_INIT)
	{
		GPI_remove_file_list (TRUE);
	}

	GPI_Init_start_time = TIME;

	/*
	 * Call a barrier in order to synchronize all tasks using MPIINIT_EV / END.
	 *  Three consecutive barriers for a better synchronization (J suggested)
	 */
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();

	initTracingTime = GPI_Init_end_time = TIME;

	Backend_postInitialize(TASKID, Extrae_get_num_tasks(), GPI_INIT_EV,
	    GPI_Init_start_time, GPI_Init_end_time, NULL);

	return ret;
}

gaspi_return_t
gaspi_proc_term(gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GPI_term_Entry();

	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_MPI_INIT)
	{
		Backend_Finalize ();
		ret = pgaspi_proc_term(timeout_ms);
		mpitrace_on = FALSE;
	} else
	{
		ret = GASPI_SUCCESS;
	}

	Extrae_GPI_term_Exit();

	return ret;
}

gaspi_return_t
gaspi_barrier(const gaspi_group_t group, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GPI_barrier_Entry();
	ret = pgaspi_barrier(group, timeout_ms);
	Extrae_GPI_barrier_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_create(const gaspi_segment_id_t segment_id,
    const gaspi_size_t size, const gaspi_group_t group,
    const gaspi_timeout_t timeout_ms, const gaspi_alloc_t alloc_policy)
{
	DBG

	int ret;

	Extrae_GPI_segment_create_Entry(segment_id, size, group);
	ret = pgaspi_segment_create(segment_id, size, group, timeout_ms, alloc_policy);
	Extrae_GPI_segment_create_Exit();

	return ret;
}

gaspi_return_t
gaspi_write(const gaspi_segment_id_t segment_id_local,
    const gaspi_offset_t offset_local, const gaspi_rank_t rank,
    const gaspi_segment_id_t segment_id_remote,
    const gaspi_offset_t offset_remote, const gaspi_size_t size,
    const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GPI_write_Entry();
	ret = pgaspi_write(segment_id_local, offset_local, rank, segment_id_remote, offset_remote, size, queue, timeout_ms);
	Extrae_GPI_write_Exit();

	return ret;
}
