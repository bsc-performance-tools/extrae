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

#include <PGASPI.h>

void	Extrae_set_trace_GASPI(int);
int		Extrae_get_trace_GASPI();

void	Extrae_set_trace_GASPI_HWC(int);
int		Extrae_get_trace_GASPI_HWC();


void	Probe_GASPI_init_Entry();
void	Probe_GASPI_init_Exit();

void	Probe_GASPI_term_Entry();
void	Probe_GASPI_term_Exit();

void	Probe_GASPI_connect_Entry();
void	Probe_GASPI_connect_Exit();

void	Probe_GASPI_disconnect_Entry();
void	Probe_GASPI_disconnect_Exit();

void	Probe_GASPI_group_create_Entry();
void	Probe_GASPI_group_create_Exit();

void	Probe_GASPI_group_add_Entry();
void	Probe_GASPI_group_add_Exit();

void	Probe_GASPI_group_commit_Entry();
void	Probe_GASPI_group_commit_Exit();

void	Probe_GASPI_group_delete_Entry();
void	Probe_GASPI_group_delete_Exit();

void	Probe_GASPI_segment_alloc_Entry(const gaspi_size_t);
void	Probe_GASPI_segment_alloc_Exit();

void	Probe_GASPI_segment_register_Entry(const gaspi_rank_t);
void	Probe_GASPI_segment_register_Exit();

void	Probe_GASPI_segment_create_Entry(const gaspi_size_t);
void	Probe_GASPI_segment_create_Exit();

void	Probe_GASPI_segment_bind_Entry(const gaspi_size_t);
void	Probe_GASPI_segment_bind_Exit();

void	Probe_GASPI_segment_use_Entry(const gaspi_size_t);
void	Probe_GASPI_segment_use_Exit();

void	Probe_GASPI_segment_delete_Entry();
void	Probe_GASPI_segment_delete_Exit();

void	Probe_GASPI_write_Entry(const gaspi_rank_t, const gaspi_size_t,
            const gaspi_queue_id_t);
void	Probe_GASPI_write_Exit();

void	Probe_GASPI_read_Entry(const gaspi_rank_t, const gaspi_size_t,
            const gaspi_queue_id_t);
void	Probe_GASPI_read_Exit();

void	Probe_GASPI_wait_Entry(const gaspi_queue_id_t);
void	Probe_GASPI_wait_Exit();

void	Probe_GASPI_notify_Entry(const gaspi_rank_t,
            const gaspi_notification_id_t, const gaspi_queue_id_t);
void	Probe_GASPI_notify_Exit();

void	Probe_GASPI_notify_waitsome_Entry(const gaspi_notification_id_t);
void	Probe_GASPI_notify_waitsome_Exit();

void	Probe_GASPI_notify_reset_Entry(const gaspi_notification_id_t);
void	Probe_GASPI_notify_reset_Exit();

void	Probe_GASPI_write_notify_Entry(const gaspi_rank_t, const gaspi_size_t,
            const gaspi_notification_id_t, const gaspi_queue_id_t);
void	Probe_GASPI_write_notify_Exit();

void	Probe_GASPI_write_list_Entry(const gaspi_rank_t, gaspi_size_t * const,
            const gaspi_queue_id_t);
void	Probe_GASPI_write_list_Exit();

void	Probe_GASPI_write_list_notify_Entry(const gaspi_rank_t,
            gaspi_size_t * const, const gaspi_notification_id_t notification_id,
            const gaspi_queue_id_t queue);
void	Probe_GASPI_write_list_notify_Exit();

void	Probe_GASPI_read_list_Entry(const gaspi_rank_t, gaspi_size_t * const,
            const gaspi_queue_id_t queue);
void	Probe_GASPI_read_list_Exit();

void	Probe_GASPI_passive_send_Entry(const gaspi_rank_t, const gaspi_size_t);
void	Probe_GASPI_passive_send_Exit();

void	Probe_GASPI_passive_receive_Entry(gaspi_rank_t * const,
            const gaspi_size_t);
void	Probe_GASPI_passive_receive_Exit();

void	Probe_GASPI_atomic_fetch_add_Entry(const gaspi_rank_t);
void	Probe_GASPI_atomic_fetch_add_Exit();

void	Probe_GASPI_atomic_compare_swap_Entry(const gaspi_rank_t);
void	Probe_GASPI_atomic_compare_swap_Exit();

void	Probe_GASPI_barrier_Entry();
void	Probe_GASPI_barrier_Exit();

void	Probe_GASPI_allreduce_Entry();
void	Probe_GASPI_allreduce_Exit();

void	Probe_GASPI_allreduce_user_Entry();
void	Probe_GASPI_allreduce_user_Exit();
