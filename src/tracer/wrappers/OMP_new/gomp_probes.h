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

#include "gomp_helpers.h"

void xtr_probe_entry_GOMP_parallel_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_start_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_start_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_static_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_static_start_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static_start_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic_start_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic_start_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_dynamic_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_dynamic_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided_start_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided_start_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_guided_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_guided_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime_start_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime_start_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_maybe_nonmonotonic_runtime_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_maybe_nonmonotonic_runtime_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_runtime_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_runtime_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_task_OL(struct task_helper_t *task_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_task_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop_OL(struct taskloop_helper_t *task_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop_ull_OL(struct taskloop_helper_t *taskloop_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop_ull_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections_start_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections_start_OL() __attribute__((weak));
void xtr_probe_entry_GOMP_teams_reg_OL(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_teams_reg_OL() __attribute__((weak));

void xtr_probe_entry_GOMP_barrier() __attribute__((weak));
void xtr_probe_exit_GOMP_barrier() __attribute__((weak));

void xtr_probe_entry_GOMP_critical_start() __attribute__((weak));
void xtr_probe_exit_GOMP_critical_start() __attribute__((weak));
void xtr_probe_entry_GOMP_critical_end() __attribute__((weak));
void xtr_probe_exit_GOMP_critical_end() __attribute__((weak));
void xtr_probe_entry_GOMP_critical_name_start(void **pptr) __attribute__((weak));
void xtr_probe_exit_GOMP_critical_name_start(void **pptr) __attribute__((weak));
void xtr_probe_entry_GOMP_critical_name_end(void **pptr) __attribute__((weak));
void xtr_probe_exit_GOMP_critical_name_end(void **pptr) __attribute__((weak));

void xtr_probe_entry_GOMP_atomic_start() __attribute__((weak));
void xtr_probe_exit_GOMP_atomic_start() __attribute__((weak));
void xtr_probe_entry_GOMP_atomic_end() __attribute__((weak));
void xtr_probe_exit_GOMP_atomic_end() __attribute__((weak));

void xtr_probe_entry_GOMP_loop_static_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_static_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_dynamic_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_dynamic_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_guided_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_guided_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_runtime_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_runtime_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_dynamic_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_dynamic_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_guided_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_guided_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_runtime_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_runtime_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_maybe_nonmonotonic_runtime_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_maybe_nonmonotonic_runtime_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_static_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_static_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_dynamic_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_dynamic_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_guided_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_guided_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_runtime_start() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_runtime_start() __attribute__((weak));

void xtr_probe_entry_GOMP_loop_static_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_static_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_dynamic_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_dynamic_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_guided_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_guided_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_runtime_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_runtime_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_dynamic_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_dynamic_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_guided_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_guided_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_runtime_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_runtime_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_maybe_nonmonotonic_runtime_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_maybe_nonmonotonic_runtime_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_static_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_static_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_dynamic_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_dynamic_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_guided_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_guided_next() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_runtime_next() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_runtime_next() __attribute__((weak));

void xtr_probe_entry_GOMP_loop_doacross_static_start(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_static_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_doacross_dynamic_start(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_dynamic_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_doacross_guided_start(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_guided_start() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_doacross_runtime_start(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_runtime_start() __attribute__((weak));

void xtr_probe_entry_GOMP_parallel_loop_static_start() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static_start(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic_start() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic_start(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided_start() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided_start(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime_start() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime_start(const void *outlined_fn) __attribute__((weak));

void xtr_probe_entry_GOMP_parallel_loop_static() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_dynamic() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_dynamic() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_guided() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_guided() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_runtime() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_runtime() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_maybe_nonmonotonic_runtime() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_maybe_nonmonotonic_runtime() __attribute__((weak));

void xtr_probe_entry_GOMP_loop_end() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_end() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_end_nowait() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_end_nowait() __attribute__((weak));

void xtr_probe_entry_GOMP_ordered_start() __attribute__((weak));
void xtr_probe_exit_GOMP_ordered_start() __attribute__((weak));
void xtr_probe_entry_GOMP_ordered_end() __attribute__((weak));
void xtr_probe_exit_GOMP_ordered_end() __attribute__((weak));
void xtr_probe_entry_GOMP_doacross_post() __attribute__((weak));
void xtr_probe_exit_GOMP_doacross_post() __attribute__((weak));
void xtr_probe_entry_GOMP_doacross_wait() __attribute__((weak));
void xtr_probe_exit_GOMP_doacross_wait() __attribute__((weak));

void xtr_probe_entry_GOMP_parallel_start() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_start(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_end() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_end() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel() __attribute__((weak));

void xtr_probe_entry_GOMP_task(struct task_helper_t *task_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_task() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop(struct taskloop_helper_t *taskloop_helper, int num_tasks) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop_ull(struct taskloop_helper_t *taskloop_helper, int num_tasks) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop_ull() __attribute__((weak));
void xtr_probe_entry_GOMP_taskwait() __attribute__((weak));
void xtr_probe_exit_GOMP_taskwait() __attribute__((weak));
void xtr_probe_entry_GOMP_taskyield() __attribute__((weak));
void xtr_probe_exit_GOMP_taskyield() __attribute__((weak));

void xtr_probe_entry_GOMP_taskgroup_start() __attribute__((weak));
void xtr_probe_exit_GOMP_taskgroup_start() __attribute__((weak));
void xtr_probe_entry_GOMP_taskgroup_end() __attribute__((weak));
void xtr_probe_exit_GOMP_taskgroup_end() __attribute__((weak));

void xtr_probe_entry_GOMP_sections_start() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_start() __attribute__((weak));
void xtr_probe_entry_GOMP_sections_next() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_next() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections_start() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections_start(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections() __attribute__((weak));
void xtr_probe_entry_GOMP_sections_end() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_end() __attribute__((weak));
void xtr_probe_entry_GOMP_sections_end_nowait() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_end_nowait() __attribute__((weak));

void xtr_probe_entry_GOMP_single_start() __attribute__((weak));
void xtr_probe_exit_GOMP_single_start() __attribute__((weak));
void xtr_probe_entry_GOMP_teams_reg() __attribute__((weak));
void xtr_probe_exit_GOMP_teams_reg() __attribute__((weak));

void xtr_probe_entry_GOMP_target() __attribute__((weak));
void xtr_probe_exit_GOMP_target() __attribute__((weak));
void xtr_probe_entry_GOMP_target_data() __attribute__((weak));
void xtr_probe_exit_GOMP_target_data() __attribute__((weak));
void xtr_probe_entry_GOMP_target_end_data() __attribute__((weak));
void xtr_probe_exit_GOMP_target_end_data() __attribute__((weak));
void xtr_probe_entry_GOMP_target_update() __attribute__((weak));
void xtr_probe_exit_GOMP_target_update() __attribute__((weak));
void xtr_probe_entry_GOMP_target_enter_exit_data() __attribute__((weak));
void xtr_probe_exit_GOMP_target_enter_exit_data() __attribute__((weak));
void xtr_probe_entry_GOMP_target_ext() __attribute__((weak));
void xtr_probe_exit_GOMP_target_ext() __attribute__((weak));
void xtr_probe_entry_GOMP_target_data_ext() __attribute__((weak));
void xtr_probe_exit_GOMP_target_data_ext() __attribute__((weak));
void xtr_probe_entry_GOMP_target_update_ext() __attribute__((weak));
void xtr_probe_exit_GOMP_target_update_ext() __attribute__((weak));

void xtr_probe_entry_omp_set_num_threads() __attribute__((weak));
void xtr_probe_exit_omp_set_num_threads(int num_threads) __attribute__((weak));
void xtr_probe_entry_omp_set_num_threads_() __attribute__((weak));
void xtr_probe_exit_omp_set_num_threads_(int num_threads) __attribute__((weak));

void xtr_probe_entry_omp_set_lock(void *lock) __attribute__((weak));
void xtr_probe_exit_omp_set_lock(void *lock) __attribute__((weak));
void xtr_probe_entry_omp_unset_lock(void *lock) __attribute__((weak));
void xtr_probe_exit_omp_unset_lock(void *lock) __attribute__((weak));

void xtr_probe_entry_omp_set_lock_( void *lock) __attribute__ ((weak));
void xtr_probe_exit_omp_set_lock_( void * lock) __attribute__ ((weak));
void xtr_probe_entry_omp_unset_lock_( void * lock) __attribute__ ((weak));
void xtr_probe_exit_omp_unset_lock_( void * lock) __attribute__ ((weak));
