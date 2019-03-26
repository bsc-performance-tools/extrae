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

gaspi_return_t  gaspi_proc_init(const gaspi_timeout_t);
gaspi_return_t  gaspi_proc_term(const gaspi_timeout_t);
gaspi_return_t  gaspi_connect(const gaspi_rank_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_disconnect(const gaspi_rank_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_group_create(gaspi_group_t * const);
gaspi_return_t  gaspi_group_add(const gaspi_group_t, const gaspi_rank_t);
gaspi_return_t  gaspi_group_commit(const gaspi_group_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_group_delete(const gaspi_group_t);
gaspi_return_t  gaspi_segment_alloc(const gaspi_segment_id_t,
                    const gaspi_size_t, const gaspi_alloc_t);
gaspi_return_t  gaspi_segment_register(const gaspi_segment_id_t,
                    const gaspi_rank_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_segment_create(const gaspi_segment_id_t,
                    const gaspi_size_t, const gaspi_group_t,
                    const gaspi_timeout_t, const gaspi_alloc_t);
gaspi_return_t  gaspi_segment_bind(const gaspi_segment_id_t,
                    const gaspi_pointer_t, const gaspi_size_t,
                    const gaspi_memory_description_t);
gaspi_return_t  gaspi_segment_use(const gaspi_segment_id_t,
                    const gaspi_pointer_t, const gaspi_size_t,
                    const gaspi_group_t, const gaspi_timeout_t,
                    const gaspi_memory_description_t);
gaspi_return_t  gaspi_segment_delete(const gaspi_segment_id_t);
gaspi_return_t  gaspi_write(const gaspi_segment_id_t, const gaspi_offset_t,
                    const gaspi_rank_t, const gaspi_segment_id_t,
                    const gaspi_offset_t, const gaspi_size_t,
                    const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_read(const gaspi_segment_id_t, const gaspi_offset_t,
                    const gaspi_rank_t, const gaspi_segment_id_t,
                    const gaspi_offset_t, const gaspi_size_t,
                    const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_wait(const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_notify(const gaspi_segment_id_t, const gaspi_rank_t,
                    const gaspi_notification_id_t, const gaspi_notification_t,
                    const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_notify_waitsome(const gaspi_segment_id_t,
                    const gaspi_notification_id_t, const gaspi_number_t,
                    gaspi_notification_id_t * const, const gaspi_timeout_t);
gaspi_return_t  gaspi_notify_reset(const gaspi_segment_id_t,
                    const gaspi_notification_id_t,
                    gaspi_notification_t * const);
gaspi_return_t  gaspi_write_notify(const gaspi_segment_id_t,
                    const gaspi_offset_t, const gaspi_rank_t,
                    const gaspi_segment_id_t, const gaspi_offset_t,
                    const gaspi_size_t, const gaspi_notification_id_t,
                    const gaspi_notification_t, const gaspi_queue_id_t,
                    const gaspi_timeout_t);
gaspi_return_t  gaspi_write_list(const gaspi_number_t,
                    gaspi_segment_id_t * const, gaspi_offset_t * const,
                    const gaspi_rank_t, gaspi_segment_id_t * const,
                    gaspi_offset_t * const, gaspi_size_t * const,
                    const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_write_list_notify(const gaspi_number_t,
                    gaspi_segment_id_t * const, gaspi_offset_t * const,
                    const gaspi_rank_t, gaspi_segment_id_t * const,
                    gaspi_offset_t * const, gaspi_size_t * const,
                    const gaspi_segment_id_t, const gaspi_notification_id_t,
                    const gaspi_notification_t, const gaspi_queue_id_t,
                    const gaspi_timeout_t);
gaspi_return_t  gaspi_read_notify(const gaspi_segment_id_t,
                    const gaspi_offset_t, const gaspi_rank_t,
                    const gaspi_segment_id_t, const gaspi_offset_t,
                    const gaspi_size_t, const gaspi_notification_id_t,
                    const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_read_list(const gaspi_number_t,
                    gaspi_segment_id_t * const, gaspi_offset_t * const,
                    const gaspi_rank_t, gaspi_segment_id_t * const,
                    gaspi_offset_t * const, gaspi_size_t * const,
                    const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_read_list_notify(const gaspi_number_t,
                    gaspi_segment_id_t * const, gaspi_offset_t * const,
                    const gaspi_rank_t, gaspi_segment_id_t * const,
                    gaspi_offset_t * const, gaspi_size_t * const,
                    const gaspi_segment_id_t, const gaspi_notification_id_t,
                    const gaspi_queue_id_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_passive_send(const gaspi_segment_id_t,
                    const gaspi_offset_t, const gaspi_rank_t,
                    const gaspi_size_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_passive_receive(const gaspi_segment_id_t,
                    const gaspi_offset_t, gaspi_rank_t * const,
                    const gaspi_size_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_atomic_fetch_add(const gaspi_segment_id_t,
                    const gaspi_offset_t, const gaspi_rank_t,
                    const gaspi_atomic_value_t, gaspi_atomic_value_t * const,
                    const gaspi_timeout_t);
gaspi_return_t  gaspi_atomic_compare_swap(const gaspi_segment_id_t,
                    const gaspi_offset_t, const gaspi_rank_t,
                    const gaspi_atomic_value_t, const gaspi_atomic_value_t,
                    gaspi_atomic_value_t * const, const gaspi_timeout_t);
gaspi_return_t  gaspi_barrier(const gaspi_group_t, const gaspi_timeout_t);
gaspi_return_t  gaspi_allreduce(gaspi_pointer_t const, gaspi_pointer_t const,
                    const gaspi_number_t, const gaspi_operation_t,
                    const gaspi_datatype_t, const gaspi_group_t,
                    const gaspi_timeout_t);
gaspi_return_t  gaspi_allreduce_user(gaspi_pointer_t const,
                    gaspi_pointer_t const, const gaspi_number_t,
                    const gaspi_size_t, const gaspi_reduce_operation_t,
                    const gaspi_reduce_state_t, const gaspi_group_t,
                    const gaspi_timeout_t);
gaspi_return_t  gaspi_queue_create(gaspi_queue_id_t * const,
                    const gaspi_timeout_t);
gaspi_return_t  gaspi_queue_delete(const gaspi_queue_id_t);
