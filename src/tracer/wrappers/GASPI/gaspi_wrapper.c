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

#include "gaspi_events.h"
#include "gaspi_wrapper.h"

static gaspi_rank_t
Extrae_GASPI_NumTasks()
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
Extrae_GASPI_TaskID()
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
Extrae_GASPI_Barrier()
{
	// XXX Change GASPI_BLOCK to actual timeout
	pgaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK);
}

static void
Extrae_GASPI_Finalize()
{
	// XXX Change GASPI_BLOCK to actual timeout
	pgaspi_proc_term(GASPI_BLOCK);
}

/*
 * GASPI_remove_file_list
 */
void
GASPI_remove_file_list(int all)
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
	iotimer_t GASPI_Init_start_time, GASPI_Init_end_time;

	ret = pgaspi_proc_init(timeout_ms);

	Extrae_set_ApplicationIsGASPI(TRUE);

	/* Setup callbacks for TASK identification and barrier execution */
	Extrae_set_taskid_function((unsigned int (*)(void))Extrae_GASPI_TaskID);
	Extrae_set_numtasks_function((unsigned int (*)(void))Extrae_GASPI_NumTasks);
	Extrae_set_barrier_tasks_function(Extrae_GASPI_Barrier);
	Extrae_set_finalize_task_function(Extrae_GASPI_Finalize);

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
		GASPI_remove_file_list (TRUE);
	}

	GASPI_Init_start_time = TIME;

	/*
	 * Call a barrier in order to synchronize all tasks using MPIINIT_EV / END.
	 *  Three consecutive barriers for a better synchronization (J suggested)
	 */
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();
	Extrae_barrier_tasks();

	initTracingTime = GASPI_Init_end_time = TIME;

	Backend_postInitialize(TASKID, Extrae_get_num_tasks(), GASPI_INIT_EV,
	    GASPI_Init_start_time, GASPI_Init_end_time, NULL);

	return ret;
}

gaspi_return_t
gaspi_proc_term(gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_term_Entry();

	if (Extrae_is_initialized_Wrapper() == EXTRAE_INITIALIZED_MPI_INIT)
	{
		Backend_Finalize ();
		ret = pgaspi_proc_term(timeout_ms);
		mpitrace_on = FALSE;
	} else
	{
		ret = GASPI_SUCCESS;
	}

	Extrae_GASPI_term_Exit();

	return ret;
}

gaspi_return_t
gaspi_connect(const gaspi_rank_t rank, const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_connect_Entry();
	ret = pgaspi_connect(rank, timeout);
	Extrae_GASPI_connect_Exit();

	return ret;
}

gaspi_return_t
gaspi_disconnect(const gaspi_rank_t rank, const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_disconnect_Entry();
	ret = pgaspi_disconnect(rank, timeout);
	Extrae_GASPI_disconnect_Exit();

	return ret;
}

gaspi_return_t
gaspi_group_create(gaspi_group_t * const group)
{
	DBG

	int ret;

	Extrae_GASPI_group_create_Entry();
	ret = pgaspi_group_create(group);
	Extrae_GASPI_group_create_Exit();

	return ret;
}

gaspi_return_t
gaspi_group_add(const gaspi_group_t group, const gaspi_rank_t rank)
{
	DBG

	int ret;

	Extrae_GASPI_group_add_Entry();
	ret = pgaspi_group_add(group, rank);
	Extrae_GASPI_group_add_Exit();

	return ret;
}

gaspi_return_t
gaspi_group_commit(const gaspi_group_t group, const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_group_commit_Entry();
	ret = pgaspi_group_commit(group, timeout);
	Extrae_GASPI_group_commit_Exit();

	return ret;
}

gaspi_return_t
gaspi_group_delete(const gaspi_group_t group)
{
	DBG

	int ret;

	Extrae_GASPI_group_delete_Entry();
	ret = pgaspi_group_delete(group);
	Extrae_GASPI_group_delete_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_alloc(const gaspi_segment_id_t segment_id,
  const gaspi_size_t size, const gaspi_alloc_t alloc_policy)
{
	DBG

	int ret;

	Extrae_GASPI_segment_alloc_Entry(size);
	ret = pgaspi_segment_alloc(segment_id, size, alloc_policy);
	Extrae_GASPI_segment_alloc_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_register(const gaspi_segment_id_t segment_id,
  const gaspi_rank_t rank, const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_segment_register_Entry();
	ret = pgaspi_segment_register(segment_id, rank, timeout);
	Extrae_GASPI_segment_register_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_create(const gaspi_segment_id_t segment_id,
    const gaspi_size_t size, const gaspi_group_t group,
    const gaspi_timeout_t timeout_ms, const gaspi_alloc_t alloc_policy)
{
	DBG

	int ret;

	Extrae_GASPI_segment_create_Entry(size);
	ret = pgaspi_segment_create(segment_id, size, group, timeout_ms, alloc_policy);
	Extrae_GASPI_segment_create_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_bind(const gaspi_segment_id_t segment_id,
    const gaspi_pointer_t pointer, const gaspi_size_t size,
    const gaspi_memory_description_t memory_description)
{
	DBG

	int ret;

	Extrae_GASPI_segment_bind_Entry(size);
	ret = pgaspi_segment_bind(segment_id, pointer, size, memory_description);
	Extrae_GASPI_segment_bind_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_use(const gaspi_segment_id_t segment_id,
    const gaspi_pointer_t pointer, const gaspi_size_t size,
    const gaspi_group_t group, const gaspi_timeout_t timeout,
    const gaspi_memory_description_t memory_description)
{
	DBG

	int ret;

	Extrae_GASPI_segment_use_Entry(size);
	ret = pgaspi_segment_use
	    (segment_id, pointer, size, group, timeout, memory_description);
	Extrae_GASPI_segment_use_Exit();

	return ret;
}

gaspi_return_t
gaspi_segment_delete(const gaspi_segment_id_t segment_id)
{
	DBG

	int ret;

	Extrae_GASPI_segment_delete_Entry();
	ret = pgaspi_segment_delete(segment_id);
	Extrae_GASPI_segment_delete_Exit();

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

	Extrae_GASPI_write_Entry(rank, size);
	ret = pgaspi_write(segment_id_local, offset_local, rank, segment_id_remote,
	    offset_remote, size, queue, timeout_ms);
	Extrae_GASPI_write_Exit();

	return ret;
}

gaspi_return_t
gaspi_read(const gaspi_segment_id_t segment_id_local,
    const gaspi_offset_t offset_local, const gaspi_rank_t rank,
    const gaspi_segment_id_t segment_id_remote,
    const gaspi_offset_t offset_remote, const gaspi_size_t size,
    const gaspi_queue_id_t queue, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_read_Entry(rank, size);
	ret = pgaspi_read(segment_id_local, offset_local, rank, segment_id_remote,
	    offset_remote, size, queue, timeout_ms);
	Extrae_GASPI_read_Exit();

	return ret;
}

gaspi_return_t
gaspi_wait(const gaspi_queue_id_t queue, const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_wait_Entry();
	ret = pgaspi_wait(queue, timeout);
	Extrae_GASPI_wait_Exit();

	return ret;
}

gaspi_return_t
gaspi_notify(const gaspi_segment_id_t segment_id,
    const gaspi_rank_t rank, const gaspi_notification_id_t notification_id,
    const gaspi_notification_t notification_value, const gaspi_queue_id_t queue,
    const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_notify_Entry(rank);
	ret = pgaspi_notify
	    (segment_id, rank, notification_id, notification_value, queue,
	    timeout);
	Extrae_GASPI_notify_Exit();

	return ret;
}

gaspi_return_t
gaspi_notify_waitsome(const gaspi_segment_id_t segment_id_local,
    const gaspi_notification_id_t notification_begin,
    const gaspi_number_t num,
    gaspi_notification_id_t * const first_id, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_notify_waitsome_Entry();
	ret = pgaspi_notify_waitsome(segment_id_local, notification_begin, num,
	    first_id, timeout_ms);
	Extrae_GASPI_notify_waitsome_Exit();

	return ret;
}

gaspi_return_t
gaspi_notify_reset(const gaspi_segment_id_t segment_id_local,
    const gaspi_notification_id_t notification_id,
    gaspi_notification_t * const old_notification_val)
{
	DBG

	int ret;

	Extrae_GASPI_notify_reset_Entry();
	ret = pgaspi_notify_reset(segment_id_local, notification_id,
	    old_notification_val);
	Extrae_GASPI_notify_reset_Exit();

	return ret;
}

gaspi_return_t
gaspi_write_notify(const gaspi_segment_id_t segment_id_local,
    const gaspi_offset_t offset_local, const gaspi_rank_t rank,
    const gaspi_segment_id_t segment_id_remote,
    const gaspi_offset_t offset_remote, const gaspi_size_t size,
    const gaspi_notification_id_t notification_id,
	const gaspi_notification_t notification_value, const gaspi_queue_id_t queue,
    const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_write_notify_Entry(rank, size);
	ret = pgaspi_write_notify(segment_id_local, offset_local, rank,
	    segment_id_remote, offset_remote, size, notification_id,
	    notification_value, queue, timeout);
	Extrae_GASPI_write_notify_Exit();

	return ret;
}

gaspi_return_t
gaspi_write_list(const gaspi_number_t num,
    gaspi_segment_id_t * const segment_id_local,
    gaspi_offset_t * const offset_local, const gaspi_rank_t rank,
    gaspi_segment_id_t * const segment_id_remote,
    gaspi_offset_t * const offset_remote, gaspi_size_t * const size,
    const gaspi_queue_id_t queue, const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_write_list_Entry(rank, size);
	ret = pgaspi_write_list(num, segment_id_local, offset_local, rank,
	    segment_id_remote, offset_remote, size, queue, timeout);
	Extrae_GASPI_write_list_Exit();

	return ret;
}

gaspi_return_t
gaspi_write_list_notify(const gaspi_number_t num,
    gaspi_segment_id_t * const segment_id_local,
    gaspi_offset_t * const offset_local, const gaspi_rank_t rank,
    gaspi_segment_id_t * const segment_id_remote,
    gaspi_offset_t * const offset_remote, gaspi_size_t * const size,
    const gaspi_segment_id_t segment_id_notification,
    const gaspi_notification_id_t notification_id,
    const gaspi_notification_t notification_value, const gaspi_queue_id_t queue,
    const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_write_list_notify_Entry(rank, size);
	ret = pgaspi_write_list_notify(num, segment_id_local, offset_local, rank,
	    segment_id_remote, offset_remote, size, segment_id_notification,
	    notification_id, notification_value, queue, timeout_ms);
	Extrae_GASPI_write_list_notify_Exit();

	return ret;
}

gaspi_return_t
gaspi_read_list(const gaspi_number_t num,
    gaspi_segment_id_t * const segment_id_local,
    gaspi_offset_t * const offset_local, const gaspi_rank_t rank,
    gaspi_segment_id_t * const segment_id_remote,
    gaspi_offset_t * const offset_remote, gaspi_size_t * const size,
    const gaspi_queue_id_t queue, const gaspi_timeout_t timeout)
{
	DBG

	int ret;

	Extrae_GASPI_read_list_Entry(rank, size);
	ret = pgaspi_read_list(num, segment_id_local, offset_local, rank,
	    segment_id_remote, offset_remote, size, queue, timeout);
	Extrae_GASPI_read_list_Exit();

	return ret;
}

gaspi_return_t
gaspi_passive_send(const gaspi_segment_id_t segment_id_local,
    const gaspi_offset_t offset_local, const gaspi_rank_t rank,
    const gaspi_size_t size, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_passive_send_Entry(rank, size);
	ret = pgaspi_passive_send(segment_id_local, offset_local, rank, size,
	    timeout_ms);
	Extrae_GASPI_passive_send_Exit();

	return ret;
}

gaspi_return_t
gaspi_passive_receive(const gaspi_segment_id_t segment_id_local,
    const gaspi_offset_t offset_local, gaspi_rank_t * const rem_rank,
    const gaspi_size_t size, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_passive_receive_Entry(rem_rank, size);
	ret = pgaspi_passive_receive(segment_id_local, offset_local, rem_rank, size,
	    timeout_ms);
	Extrae_GASPI_passive_receive_Exit();

	return ret;
}

gaspi_return_t
gaspi_atomic_fetch_add(const gaspi_segment_id_t segment_id,
    const gaspi_offset_t offset, const gaspi_rank_t rank,
    const gaspi_atomic_value_t val_add, gaspi_atomic_value_t * const val_old,
    const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_atomic_fetch_add_Entry(rank);
	ret = pgaspi_atomic_fetch_add(segment_id, offset, rank, val_add, val_old,
	    timeout_ms);
	Extrae_GASPI_atomic_fetch_add_Exit();

	return ret;
}

gaspi_return_t
gaspi_atomic_compare_swap(const gaspi_segment_id_t segment_id,
    const gaspi_offset_t offset, const gaspi_rank_t rank,
    const gaspi_atomic_value_t comparator, const gaspi_atomic_value_t val_new,
    gaspi_atomic_value_t * const val_old, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_atomic_compare_swap_Entry(rank);
	ret = pgaspi_atomic_compare_swap(segment_id, offset, rank, comparator,
	    val_new, val_old, timeout_ms);
	Extrae_GASPI_atomic_compare_swap_Exit();

	return ret;
}

gaspi_return_t
gaspi_barrier(const gaspi_group_t group, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_barrier_Entry();
	ret = pgaspi_barrier(group, timeout_ms);
	Extrae_GASPI_barrier_Exit();

	return ret;
}

gaspi_return_t
gaspi_allreduce(gaspi_pointer_t const buffer_send,
    gaspi_pointer_t const buffer_receive, const gaspi_number_t num,
    const gaspi_operation_t operation, const gaspi_datatype_t datatyp,
    const gaspi_group_t group, const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_allreduce_Entry();
	ret = pgaspi_allreduce(buffer_send, buffer_receive, num, operation, datatyp,
	    group, timeout_ms);
	Extrae_GASPI_allreduce_Exit();

	return ret;
}

gaspi_return_t
gaspi_allreduce_user(gaspi_pointer_t const buffer_send,
    gaspi_pointer_t const buffer_receive, const gaspi_number_t num,
    const gaspi_size_t element_size, const gaspi_reduce_operation_t operation,
    const gaspi_state_t reduce_state, const gaspi_group_t group,
    const gaspi_timeout_t timeout_ms)
{
	DBG

	int ret;

	Extrae_GASPI_allreduce_user_Entry();
	ret = pgaspi_allreduce_user(buffer_send, buffer_receive, num, element_size,
	    operation, reduce_state, group, timeout_ms);
	Extrae_GASPI_allreduce_user_Exit();

	return ret;
}
