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

void xtr_probe_entry_GOMP_parallel_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_start_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_start_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_static_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_static_start_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static_start_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic_start_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic_start_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_dynamic_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_dynamic_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided_start_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided_start_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_guided_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_guided_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime_start_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime_start_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_maybe_nonmonotonic_runtime_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_maybe_nonmonotonic_runtime_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_runtime_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_runtime_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_task_OL_bursts(struct task_helper_t *task_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_task_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop_OL_bursts(struct taskloop_helper_t *task_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop_ull_OL_bursts(struct taskloop_helper_t *taskloop_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop_ull_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections_start_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections_start_OL_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_teams_reg_OL_bursts(struct parallel_helper_t *par_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_teams_reg_OL_bursts() __attribute__((weak));

void xtr_probe_entry_GOMP_barrier_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_barrier_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_critical_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_critical_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_critical_end_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_critical_end_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_critical_name_start_bursts(void **pptr) __attribute__((weak));
void xtr_probe_exit_GOMP_critical_name_start_bursts(void **pptr) __attribute__((weak));
void xtr_probe_entry_GOMP_critical_name_end_bursts(void **pptr) __attribute__((weak));
void xtr_probe_exit_GOMP_critical_name_end_bursts(void **pptr) __attribute__((weak));
void xtr_probe_entry_GOMP_atomic_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_atomic_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_atomic_end_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_atomic_end_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_static_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_static_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_guided_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_guided_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_runtime_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_runtime_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_guided_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_guided_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_runtime_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_runtime_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_maybe_nonmonotonic_runtime_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_maybe_nonmonotonic_runtime_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_static_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_static_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_guided_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_guided_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_runtime_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_runtime_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_static_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_static_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_dynamic_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_dynamic_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_guided_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_guided_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_runtime_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_runtime_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_dynamic_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_dynamic_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_guided_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_guided_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_nonmonotonic_runtime_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_nonmonotonic_runtime_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_maybe_nonmonotonic_runtime_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_maybe_nonmonotonic_runtime_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_static_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_static_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_dynamic_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_dynamic_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_guided_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_guided_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_ordered_runtime_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_ordered_runtime_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_doacross_static_start_bursts(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_static_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_doacross_dynamic_start_bursts(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_doacross_guided_start_bursts(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_guided_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_doacross_runtime_start_bursts(unsigned ncounts) __attribute__((weak));
void xtr_probe_exit_GOMP_loop_doacross_runtime_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_static_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static_start_bursts(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic_start_bursts(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided_start_bursts(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime_start_bursts(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_static_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_static_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_dynamic_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_dynamic_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_guided_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_guided_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_runtime_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_runtime_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_dynamic_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_dynamic_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_guided_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_guided_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_nonmonotonic_runtime_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_nonmonotonic_runtime_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_loop_maybe_nonmonotonic_runtime_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_loop_maybe_nonmonotonic_runtime_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_end_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_end_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_loop_end_nowait_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_loop_end_nowait_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_ordered_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_ordered_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_ordered_end_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_ordered_end_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_doacross_post_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_doacross_post_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_doacross_wait_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_doacross_wait_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_start_bursts(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_end_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_end_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_task_bursts(struct task_helper_t *task_helper) __attribute__((weak));
void xtr_probe_exit_GOMP_task_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop_bursts(struct taskloop_helper_t *taskloop_helper, int num_tasks) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskloop_ull_bursts(struct taskloop_helper_t *taskloop_helper, int num_tasks) __attribute__((weak));
void xtr_probe_exit_GOMP_taskloop_ull_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskwait_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_taskwait_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskyield_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_taskyield_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskgroup_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_taskgroup_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_taskgroup_end_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_taskgroup_end_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_sections_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_sections_next_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_next_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections_start_bursts(const void *outlined_fn) __attribute__((weak));
void xtr_probe_entry_GOMP_parallel_sections_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_parallel_sections_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_sections_end_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_end_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_sections_end_nowait_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_sections_end_nowait_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_single_start_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_single_start_bursts() __attribute__((weak));
void xtr_probe_entry_GOMP_teams_reg_bursts() __attribute__((weak));
void xtr_probe_exit_GOMP_teams_reg_bursts() __attribute__((weak));

void xtr_probe_entry_omp_set_num_threads_bursts() __attribute__((weak));
void xtr_probe_exit_omp_set_num_threads_bursts(int num_threads) __attribute__((weak));
void xtr_probe_entry_omp_set_num_threads__bursts() __attribute__((weak));
void xtr_probe_exit_omp_set_num_threads__bursts(int num_threads) __attribute__((weak));

void xtr_probe_entry_omp_set_lock_bursts(void *lock) __attribute__((weak));
void xtr_probe_exit_omp_set_lock_bursts(void *lock) __attribute__((weak));
void xtr_probe_entry_omp_unset_lock_bursts(void *lock) __attribute__((weak));
void xtr_probe_exit_omp_unset_lock_bursts(void *lock) __attribute__((weak));

void xtr_probe_entry_omp_set_lock__bursts( void *lock) __attribute__ ((weak));
void xtr_probe_exit_omp_set_lock__bursts( void * lock) __attribute__ ((weak));
void xtr_probe_entry_omp_unset_lock__bursts( void * lock) __attribute__ ((weak));
void xtr_probe_exit_omp_unset_lock__bursts( void * lock) __attribute__ ((weak));
