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

#include "gaspi_events.h"
#include "gaspi_probe.h"
#include "wrapper.h"

void
Extrae_GASPI_init_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_init_Entry();
}

void
Extrae_GASPI_init_Exit()
{
	Probe_GASPI_init_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_term_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_term_Entry();
}

void
Extrae_GASPI_term_Exit()
{
	Probe_GASPI_term_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_connect_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_connect_Entry();
}

void
Extrae_GASPI_connect_Exit()
{
	Probe_GASPI_connect_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_disconnect_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_disconnect_Entry();
}

void
Extrae_GASPI_disconnect_Exit()
{
	Probe_GASPI_disconnect_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_group_create_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_group_create_Entry();
}

void
Extrae_GASPI_group_create_Exit()
{
	Probe_GASPI_group_create_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_group_add_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_group_add_Entry();
}

void
Extrae_GASPI_group_add_Exit()
{
	Probe_GASPI_group_add_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_group_commit_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_group_commit_Entry();
}

void
Extrae_GASPI_group_commit_Exit()
{
	Probe_GASPI_group_commit_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_group_delete_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_group_delete_Entry();
}

void
Extrae_GASPI_group_delete_Exit()
{
	Probe_GASPI_group_delete_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_segment_alloc_Entry(const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_segment_alloc_Entry(size);
}

void
Extrae_GASPI_segment_alloc_Exit()
{
	Probe_GASPI_segment_alloc_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_segment_register_Entry(const gaspi_rank_t rank)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_segment_register_Entry(rank);
}

void
Extrae_GASPI_segment_register_Exit()
{
	Probe_GASPI_segment_register_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_segment_create_Entry(const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_segment_create_Entry(size);
}

void
Extrae_GASPI_segment_create_Exit()
{
	Probe_GASPI_segment_create_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_segment_bind_Entry(const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_segment_bind_Entry(size);
}

void
Extrae_GASPI_segment_bind_Exit()
{
	Probe_GASPI_segment_bind_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_segment_use_Entry(const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_segment_use_Entry(size);
}

void
Extrae_GASPI_segment_use_Exit()
{
	Probe_GASPI_segment_use_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_segment_delete_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_segment_delete_Entry();
}

void
Extrae_GASPI_segment_delete_Exit()
{
	Probe_GASPI_segment_delete_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_write_Entry(const gaspi_rank_t rank, const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_write_Entry(rank, size);
}

void
Extrae_GASPI_write_Exit()
{
	Probe_GASPI_write_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_read_Entry(const gaspi_rank_t rank, const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_read_Entry(rank, size);
}

void
Extrae_GASPI_read_Exit()
{
	Probe_GASPI_read_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_wait_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_wait_Entry();
}

void
Extrae_GASPI_wait_Exit()
{
	Probe_GASPI_wait_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_notify_Entry(const gaspi_rank_t rank)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_notify_Entry(rank);
}

void
Extrae_GASPI_notify_Exit()
{
	Probe_GASPI_notify_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_notify_waitsome_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_notify_waitsome_Entry();
}

void
Extrae_GASPI_notify_waitsome_Exit()
{
	Probe_GASPI_notify_waitsome_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_notify_reset_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_notify_reset_Entry();
}

void
Extrae_GASPI_notify_reset_Exit()
{
	Probe_GASPI_notify_reset_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_write_notify_Entry(const gaspi_rank_t rank, const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_write_notify_Entry(rank, size);
}

void
Extrae_GASPI_write_notify_Exit()
{
	Probe_GASPI_write_notify_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_write_list_Entry(const gaspi_rank_t rank, gaspi_size_t * const size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_write_list_Entry(rank, size);
}

void
Extrae_GASPI_write_list_Exit()
{
	Probe_GASPI_write_list_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_write_list_notify_Entry(const gaspi_rank_t rank,
    gaspi_size_t * const size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_write_list_notify_Entry(rank, size);
}

void
Extrae_GASPI_write_list_notify_Exit()
{
	Probe_GASPI_write_list_notify_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_read_list_Entry(const gaspi_rank_t rank, gaspi_size_t * const size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_read_list_Entry(rank, size);
}

void
Extrae_GASPI_read_list_Exit()
{
	Probe_GASPI_read_list_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_passive_send_Entry(const gaspi_rank_t rank, const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_passive_send_Entry(rank, size);
}

void
Extrae_GASPI_passive_send_Exit()
{
	Probe_GASPI_passive_send_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_passive_receive_Entry(gaspi_rank_t * const rem_rank,
    const gaspi_size_t size)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_passive_receive_Entry(rem_rank, size);
}

void
Extrae_GASPI_passive_receive_Exit()
{
	Probe_GASPI_passive_receive_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_atomic_fetch_add_Entry(const gaspi_rank_t rank)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_atomic_fetch_add_Entry(rank);
}

void
Extrae_GASPI_atomic_fetch_add_Exit()
{
	Probe_GASPI_atomic_fetch_add_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_atomic_compare_swap_Entry(const gaspi_rank_t rank)
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_atomic_compare_swap_Entry(rank);
}

void
Extrae_GASPI_atomic_compare_swap_Exit()
{
	Probe_GASPI_atomic_compare_swap_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_barrier_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_barrier_Entry();
}

void
Extrae_GASPI_barrier_Exit()
{
	Probe_GASPI_barrier_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_allreduce_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_allreduce_Entry();
}

void
Extrae_GASPI_allreduce_Exit()
{
	Probe_GASPI_allreduce_Exit();
	Backend_Leave_Instrumentation();
}

void
Extrae_GASPI_allreduce_user_Entry()
{
	Backend_Enter_Instrumentation();
	Probe_GASPI_allreduce_user_Entry();
}

void
Extrae_GASPI_allreduce_user_Exit()
{
	Probe_GASPI_allreduce_user_Exit();
	Backend_Leave_Instrumentation();
}
