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

#include "wrap_macros.h"

#define GOMP_API_3_1 3.1
#define GOMP_API_4_0 4.0
#define GOMP_API_4_5 4.5
#define GOMP_API_5_0 5.0
#define GOMP_API_5_1 5.1
#define GOMP_API_5_2 5.2

void xtr_GOMP_extra_debug(char *buffer, int buffer_size) __attribute__((weak));

int xtr_OMP_GOMP_init (void);

extern float __GOMP_version;

/* libgomp/barrier.c */

extern void (*REAL_SYMBOL_PTR(GOMP_barrier)) (void);

/* libgomp/critical.c */

extern void (*REAL_SYMBOL_PTR(GOMP_critical_start)) (void);
extern void (*REAL_SYMBOL_PTR(GOMP_critical_end)) (void);
extern void (*REAL_SYMBOL_PTR(GOMP_critical_name_start)) (void **);
extern void (*REAL_SYMBOL_PTR(GOMP_critical_name_end)) (void **);
extern void (*REAL_SYMBOL_PTR(GOMP_atomic_start)) (void);
extern void (*REAL_SYMBOL_PTR(GOMP_atomic_end)) (void);

/* libgomp/icv.c*/

extern void (*REAL_SYMBOL_PTR(omp_set_num_threads)) (int);

/* Fortran-mangled function */

extern void (*REAL_SYMBOL_PTR(omp_set_num_threads_)) (int);

/* libgomp/lock.c */

extern void (*REAL_SYMBOL_PTR(omp_set_lock)) (omp_lock_t *);
extern void (*REAL_SYMBOL_PTR(omp_unset_lock)) (omp_lock_t *);

/* Fortran-mangled functions */

extern void (*REAL_SYMBOL_PTR(omp_set_lock_)) (omp_lock_t *);
extern void (*REAL_SYMBOL_PTR(omp_unset_lock_)) (omp_lock_t *);

/* libgomp/loop.c */

extern int (*REAL_SYMBOL_PTR(GOMP_loop_static_start)) (long, long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_dynamic_start)) (long, long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_guided_start)) (long, long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_runtime_start)) (long, long, long, long, long *, long *);

extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_static_start)) (long, long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_dynamic_start)) (long, long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_guided_start)) (long, long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_runtime_start)) (long, long, long, long, long *, long *);

extern int (*REAL_SYMBOL_PTR(GOMP_loop_static_next)) (long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_dynamic_next)) (long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_guided_next)) (long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_runtime_next)) (long *, long *);

extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_static_next)) (long *, long *); 
extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_dynamic_next)) (long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_guided_next)) (long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_ordered_runtime_next)) (long *, long *);

extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_static_start)) (void*, void*, unsigned, long, long, long, long);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_dynamic_start)) (void*, void*, unsigned, long, long, long, long);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_guided_start)) (void*, void*, unsigned, long, long, long, long);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_runtime_start)) (void*, void*, unsigned, long, long, long);

extern void (*REAL_SYMBOL_PTR(GOMP_loop_end)) (void);
extern void (*REAL_SYMBOL_PTR(GOMP_loop_end_nowait)) (void);

/* libgomp/ordered.c */

extern void (*REAL_SYMBOL_PTR(GOMP_ordered_start)) (void);
extern void (*REAL_SYMBOL_PTR(GOMP_ordered_end)) (void);

/* libgomp/parallel.c */

extern void (*REAL_SYMBOL_PTR(GOMP_parallel_start)) (void *, void *, unsigned);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_end)) (void);

/* libgomp/sections.c */

extern unsigned (*REAL_SYMBOL_PTR(GOMP_sections_start)) (unsigned);
extern unsigned (*REAL_SYMBOL_PTR(GOMP_sections_next)) (void);

extern void (*REAL_SYMBOL_PTR(GOMP_parallel_sections_start)) (void *, void *, unsigned, unsigned);
extern void (*REAL_SYMBOL_PTR(GOMP_sections_end)) (void);
extern void (*REAL_SYMBOL_PTR(GOMP_sections_end_nowait)) (void);

/* libgomp/single.c */

extern unsigned (*REAL_SYMBOL_PTR(GOMP_single_start)) (void);


/*********************************/
/***** Available since GCC 4.4 ***/
/*********************************/

/* libgomp/team.c, libgomp/task.c */

// GOMP_task appeared in GCC 4.4 but increased the parameters in 4.9 and later in 6.0
extern void (*REAL_SYMBOL_PTR(GOMP_task)) (void *, void *, void *, long, long, int, unsigned, ...);
extern void (*REAL_SYMBOL_PTR(GOMP_taskwait)) (void);


/*********************************/
/***** Available since GCC 4.9 ***/
/*********************************/

/* libgomp/loop.c */

extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_static)) (void*, void*, unsigned, long, long, long, long, unsigned);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_dynamic)) (void*, void*, unsigned, long, long, long, long, unsigned);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_guided)) (void*, void*, unsigned, long, long, long, long, unsigned);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_runtime)) (void*, void*, unsigned, long, long, long, unsigned);

/* libgomp/task.c */

extern void (*REAL_SYMBOL_PTR(GOMP_taskgroup_start)) (void);
extern void (*REAL_SYMBOL_PTR(GOMP_taskgroup_end)) (void);

/* libgomp/parallel.c */

extern void (*REAL_SYMBOL_PTR(GOMP_parallel)) (void *, void *, unsigned, unsigned);

/* libgomp/sections.c */

extern void (*REAL_SYMBOL_PTR(GOMP_parallel_sections)) (void *, void *, unsigned, unsigned, unsigned);


/*********************************/
/***** Available since GCC 6.0 ***/
/*********************************/

/* libgomp/loop.c */

extern int (*REAL_SYMBOL_PTR(GOMP_loop_nonmonotonic_dynamic_start)) (long, long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_nonmonotonic_guided_start)) (long, long, long, long, long *, long *);

extern int (*REAL_SYMBOL_PTR(GOMP_loop_nonmonotonic_dynamic_next)) (long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_nonmonotonic_guided_next)) (long *, long *);

extern int (*REAL_SYMBOL_PTR(GOMP_loop_doacross_static_start)) (unsigned, long *, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_doacross_dynamic_start)) (unsigned, long *, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_doacross_guided_start)) (unsigned, long *, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_doacross_runtime_start)) (unsigned, long *, long *, long *);

extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_nonmonotonic_dynamic)) (void *, void *, unsigned, long, long, long, long, unsigned);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_nonmonotonic_guided)) (void *, void *, unsigned, long, long, long, long, unsigned);

/* libgomp/ordered.c */

extern void (*REAL_SYMBOL_PTR(GOMP_doacross_post)) (long *);
extern void (*REAL_SYMBOL_PTR(GOMP_doacross_wait)) (long, ...);

/* libgomp/taskloop.c */

extern void (*REAL_SYMBOL_PTR(GOMP_taskloop)) (void *, void *, void *, long, long, unsigned, unsigned long, int, long, long, long);


/*********************************/
/***** Available since GCC 9.0 ***/
/*********************************/

/* libgomp/loop.c */

extern int (*REAL_SYMBOL_PTR(GOMP_loop_nonmonotonic_runtime_start)) (long, long, long, long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_maybe_nonmonotonic_runtime_start)) (long, long, long, long *, long *);

extern int (*REAL_SYMBOL_PTR(GOMP_loop_nonmonotonic_runtime_next)) (long *, long *);
extern int (*REAL_SYMBOL_PTR(GOMP_loop_maybe_nonmonotonic_runtime_next)) (long *, long *);

extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_nonmonotonic_runtime)) (void *, void *, unsigned, long, long, long, unsigned);
extern void (*REAL_SYMBOL_PTR(GOMP_parallel_loop_maybe_nonmonotonic_runtime)) (void *, void *, unsigned, long, long, long, unsigned);

/* libgomp/taskloop.c */

extern void (*REAL_SYMBOL_PTR(GOMP_taskloop_ull)) (void *, void *, void *, long, long, unsigned, unsigned long, int, unsigned long long, unsigned long long, unsigned long long);

/* libgomp/teams.c */

extern void (*REAL_SYMBOL_PTR(GOMP_teams_reg)) (void *, void *, unsigned int, unsigned int, unsigned int);
