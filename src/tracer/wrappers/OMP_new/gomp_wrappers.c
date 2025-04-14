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

/*
 * GCC flags to see compiler's code transformations:
 *   -fdump-tree-ompexp
 *   -fdump-tree-omplower
 *   -fdump-tree-gimple
 *
 * GNU libgomp documentation explaining compiler code transformations for
 * different constructs (Section 9, The libgomp ABI -> Implementing...):
 *
 * https://gcc.gnu.org/onlinedocs/libgomp
 */

 #if !defined(_GNU_SOURCE)
 #define _GNU_SOURCE
 #endif

#include "atomic_counter.h"
#include "gnu_libgomp.h"
#include "gomp_helpers.h"
#include "gomp_probes.h"
#include "gomp_probes_burst.h"
#include "omp_wrap_macros.h"

/**********************/
/* Outlined callbacks */
/**********************/

/**
 * GOMP_parallel_OL
 *
 * Wrapper for outlined routines started from:
 *   GOMP_parallel
 *   GOMP_parallel_start
 */
XTR_WRAP_GOMP_PARALLEL_OL(GOMP_parallel_OL);
void GOMP_parallel_start_OL( void * ) __attribute__ ((alias( "GOMP_parallel_OL" )));

/**
 * GOMP_parallel_loop_static_OL
 *
 * Wrapper for outlined routines started from:
 *   GOMP_parallel_loop_static
 *   GOMP_parallel_loop_static_start
 */
XTR_WRAP_GOMP_PARALLEL_OL(GOMP_parallel_loop_static_OL);
void GOMP_parallel_loop_static_start_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_static_OL" )));

/**
 * GOMP_parallel_loop_dynamic_OL
 *
 * Wrapper for outlined routines started from:
 *   GOMP_parallel_loop_dynamic
 *   GOMP_parallel_loop_dynamic_start
 *   GOMP_parallel_loop_nonmonotonic_dynamic
 */
XTR_WRAP_GOMP_PARALLEL_OL(GOMP_parallel_loop_dynamic_OL);
void GOMP_parallel_loop_dynamic_start_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_dynamic_OL" )));
void GOMP_parallel_loop_nonmonotonic_dynamic_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_dynamic_OL" )));

/**
 * GOMP_parallel_loop_guided_OL
 *
 * Wrapper for outlined routines started from:
 *   GOMP_parallel_loop_guided
 *   GOMP_parallel_loop_guided_start
 *   GOMP_parallel_loop_nonmonotonic_guided
 */
XTR_WRAP_GOMP_PARALLEL_OL(GOMP_parallel_loop_guided_OL);
void GOMP_parallel_loop_guided_start_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_guided_OL" )));
void GOMP_parallel_loop_nonmonotonic_guided_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_guided_OL" )));

/**
 * GOMP_parallel_loop_runtime_OL
 *
 * Wrapper for outlined routines started from:
 *   GOMP_parallel_loop_runtime_start
 *   GOMP_parallel_loop_runtime
 *   GOMP_parallel_loop_nonmonotonic_runtime
 *   GOMP_parallel_loop_maybe_nonmonotonic_runtime
 */
XTR_WRAP_GOMP_PARALLEL_OL(GOMP_parallel_loop_runtime_OL);
void GOMP_parallel_loop_runtime_start_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_runtime_OL" )));
void GOMP_parallel_loop_nonmonotonic_runtime_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_runtime_OL" )));
void GOMP_parallel_loop_maybe_nonmonotonic_runtime_OL( void * ) __attribute__ ((alias( "GOMP_parallel_loop_runtime_OL" )));

/**
 * GOMP_parallel_sections_OL
 *
 * Wrapper for outlined routines started from:
 *   GOMP_parallel_sections_start
 *   GOMP_parallel_sections
 */
XTR_WRAP_GOMP_PARALLEL_OL(GOMP_parallel_sections_OL);
void GOMP_parallel_sections_start_OL( void * ) __attribute__ ((alias( "GOMP_parallel_sections_OL" )));

/**
 * GOMP_teams_reg_OL
 *
 * Wrapper for outlined routines started from:
 *   GOMP_teams_reg
 */
XTR_WRAP_GOMP_TEAMS_OL(GOMP_teams_reg_OL);

/**
 * GOMP_task_OL
 *
 * Wrapper for outlined tasks started from:
 *   GOMP_task
 *   GOMP_taskloop
 */
XTR_WRAP_GOMP_TASK_OL(GOMP_task_OL);

XTR_WRAP_GOMP_TASKLOOP_OL(GOMP_taskloop_OL);
void GOMP_taskloop_ull_OL( void * ) __attribute__ ((alias( "GOMP_taskloop_OL" )));

/*********************/
/* libgomp/barrier.c */
/*********************/

/**
 * GOMP_barrier
 * Avail: GCC >= 4.2
 * 
 * Wrapper for an explicit barrier construct.
 * 
 * Code transformation:
 * <<<
 *    #pragma omp barrier 
 * >>>
 *    GOMP_barrier();
 */
XTR_WRAP_GOMP(GOMP_barrier,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**********************/
/* libgomp/critical.c */
/**********************/

/**
 * GOMP_critical_start
 * Avail: GCC >= 4.2
 * 
 * Wrapper for a critical construct that restricts execution of the associated 
 * structured block to a single thread at a time.
 *  
 * Code transformation:
 * <<<
 *    #pragma omp critical 
 *    {
 *      count ++;
 *    }
 * >>>
 *    GOMP_critical_start();
 *    count ++;
 *    GOMP_critical_end();
 */
XTR_WRAP_GOMP(GOMP_critical_start,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_critical_end
 * Avail: GCC >= 4.2
 * 
 * See GOMP_critical_start for details.
 */
XTR_WRAP_GOMP(GOMP_critical_end,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_critical_name_start
 * Avail: GCC >= 4.2
 * 
 * Wrapper for a critical construct with a [(name)] clause that restricts execution 
 * of the associated structured block until no other thread is executing a critical
 * region with the same name.
 * 
 * @param pptr Pointer to the mutex for the critical region. pptr (void **) distinguishes criticals with different names, 
 *             and shares the same value for those that share the same name. Dereferencing pptr (*pptr, void *), is not 
 *             useful to distinguish different criticals. 
 * 
 * Code transformation:
 * <<<
 *    #pragma omp critical(NAME) 
 *    {
 *      count ++;
 *    }
 * >>>
 *    GOMP_critical_name_start(&.gomp_critical_user_NAME);
 *    count ++;
 *    GOMP_critical_name_end(&.gomp_critical_user_NAME);
 */
XTR_WRAP_GOMP(GOMP_critical_name_start,
              PROTOTYPE(void **pptr),
              NO_RETURN,
              ENTRY_PROBE_ARGS(pptr),
              REAL_SYMBOL_ARGS(pptr),
              EXIT_PROBE_ARGS(pptr),
              DEBUG_ARGS("pptr=%p", pptr));

/**
 * GOMP_critical_name_end
 * Avail: GCC >= 4.2
 * 
 * @param pptr Pointer to the mutex for the critical region. See GOMP_critical_name_start for details.
 */
XTR_WRAP_GOMP(GOMP_critical_name_end,
              PROTOTYPE(void **pptr),
              NO_RETURN,
              ENTRY_PROBE_ARGS(pptr),
              REAL_SYMBOL_ARGS(pptr),
              EXIT_PROBE_ARGS(pptr),
              DEBUG_ARGS("pptr=%p", pptr));

/**
 * GOMP_atomic_start
 * Avail: GCC >= 4.2
 * 
 * Wrapper for the atomic construct.
 * 
 * Code transformation:
 * <<< 
 *    #pragma omp atomic  
 *    {
 *       count ++;
 *    }
 *  ** Actually, you need to use  __builtin_GOMP_atomic_start() and __builtin_GOMP_atomic_end() in the program for this wrapper to trigger.
 * >>>
 *    GOMP_atomic_start();
 *    count ++;
 *    GOMP_atomic_end();
 */
XTR_WRAP_GOMP(GOMP_atomic_start,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_atomic_end
 * Avail: GCC >= 4.2
 * 
 * See GOMP_atomic_start for details.
 */
XTR_WRAP_GOMP(GOMP_atomic_end,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/******************/
/* libgomp/loop.c */
/******************/

/**
 * GOMP_loop_static_start
 * Avail: GCC >= 4.2
 * 
 * Unlike with the other schedulings, this wrapper is not called when a '#pragma omp for scheduling(static)'
 * is found, the compiler seems to "unroll" the loop work statically and this is never issued. 
 * Code transformation:
 * <<<
 *    #pragma omp for schedule(static)
 *    for (i=0; i<NITERS; i++)
 *    {
 *       foo();
 *    }
 * >>>
 *    foo();
 *    GOMP_barrier();
 */
XTR_WRAP_GOMP(GOMP_loop_static_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_dynamic_start
 * Avail: GCC >= 4.2
 *
 * Wrapper generated for the loop construct '#pragma omp for schedule(dynamic)' when appears alone,
 * either inside or outside a parallel region, not for the combined construct '#pragma omp parallel for'
 *
 * Code transformation:
 * <<<
 *    #pragma omp for schedule(dynamic)
 *    for (i=0; i<NITERS; i++)
 *    {
 *       foo();
 *    }
 * >>>
 *    for(chunk = GOMP_loop_dynamic_start(); chunk != NULL; chunk = GOMP_loop_dynamic_next())
 *    {
 *      foo(); 
 *    }
 *    GOMP_loop_end();
 */
XTR_WRAP_GOMP(GOMP_loop_dynamic_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_guided_start
 * Avail: GCC >= 4.2
 *
 * Wrapper generated for the loop construct '#pragma omp for schedule(guided)' when appears alone,
 * either inside or outside a parallel region, not for the combined construct '#pragma omp parallel for'
 *
 * Code transformation:
 * <<<
 *    #pragma omp for schedule(guided)
 *    for (i=0; i<NITERS; i++)
 *    {
 *       foo();
 *    }
 * >>>
 *    chunk = GOMP_loop_guided_start();
 *    do
 *    {
 *      while(chunk) { foo(); i++ }
 *      chunk = GOMP_loop_guided_next();
 *    } while(chunk);
 *    GOMP_loop_end();
 */
XTR_WRAP_GOMP(GOMP_loop_guided_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_runtime_start
 * Avail: GCC >= 4.2
 *
 * Wrapper generated for the loop construct '#pragma omp for schedule(runtime)' when appears alone,
 * either inside or outside a parallel region, not for the combined construct '#pragma omp parallel for'
 *
 * Code transformation:
 * <<<
 *    #pragma omp for schedule(runtime)
 *    for (i=0; i<NITERS; i++)
 *    {
 *       foo();
 *    }
 * >>>
 *    chunk = GOMP_loop_runtime_start();
 *    do
 *    {
 *      while(chunk) { foo(); i++ }
 *      chunk = GOMP_loop_runtime_next();
 *    } while(chunk);
 *    GOMP_loop_end();
 * 
 */
XTR_WRAP_GOMP(GOMP_loop_runtime_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_nonmonotonic_dynamic_start
 * Avail: GCC >= 6
 * 
 * Replaces GOMP_loop_dynamic_start starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_nonmonotonic_dynamic_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_nonmonotonic_guided_start
 * Avail: GCC >= 6
 * 
 * Replaces GOMP_loop_guided_start starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_nonmonotonic_guided_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_nonmonotonic_runtime_start
 * Avail: GCC >= 9
 * 
 * Replaces GOMP_loop_runtime_start starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_nonmonotonic_runtime_start,
              PROTOTYPE(long start, long end, long incr, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld istart=%p iend=%p", start, end, incr, istart, iend));

/**
 * GOMP_loop_maybe_nonmonotonic_runtime_start
 * Avail: GCC >= 9
 * 
 * Replaces GOMP_loop_runtime_start starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_maybe_nonmonotonic_runtime_start,
              PROTOTYPE(long start, long end, long incr, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld istart=%p iend=%p", start, end, incr, istart, iend));

/**
 * GOMP_loop_ordered_static_start
 * Avail: GCC >= 4.2
 * 
 * Replaces GOMP_loop_static_start when the loop construct includes an 'ordered' clause. 
 * Likewise, this wrapper triggers when the loop appears either inside or outside a
 * parallel, and also in combined constructs: #pragma omp parallel for ordered schedule(static) 
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered schedule(static)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    chunk = GOMP_loop_ordered_static_start();
 *    do
 *    {
 *      while(chunk) { foo(); i++ }
 *      chunk = GOMP_loop_ordered_static_next();
 *    } while(chunk);
 *    GOMP_loop_end(); 
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_static_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_ordered_dynamic_start
 * Avail: GCC >= 4.2
 * 
 * Replaces GOMP_loop_dynamic_start when the loop construct includes an 'ordered' clause. 
 * Likewise, this wrapper triggers when the loop appears either inside or outside a
 * parallel, and also in combined constructs: #pragma omp parallel for ordered schedule(dynamic) 
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered schedule(dynamic)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    chunk = GOMP_loop_ordered_dynamic_start();
 *    do
 *    {
 *      while(chunk) { foo(); i++ }
 *      chunk = GOMP_loop_ordered_dynamic_next();
 *    } while(chunk);
 *    GOMP_loop_end(); 
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_dynamic_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_ordered_guided_start
 * Avail: GCC >= 4.2
 * 
 * Replaces GOMP_loop_guided_start when the loop construct includes an 'ordered' clause. 
 * Likewise, this wrapper triggers when the loop appears either inside or outside a
 * parallel, and also in combined constructs: #pragma omp parallel for ordered schedule(guided) 
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered schedule(guided)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    chunk = GOMP_loop_ordered_guided_start();
 *    do
 *    {
 *      while(chunk) { foo(); i++ }
 *      chunk = GOMP_loop_ordered_guided_next();
 *    } while(chunk);
 *    GOMP_loop_end(); 
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_guided_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_ordered_runtime_start
 * Avail: GCC >= 4.2
 * 
 * Replaces GOMP_loop_runtime_start when the loop construct includes an 'ordered' clause. 
 * Likewise, this wrapper triggers when the loop appears either inside or outside a
 * parallel, and also in combined constructs: #pragma omp parallel for ordered schedule(runtime) 
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered schedule(runtime)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    chunk = GOMP_loop_ordered_runtime_start();
 *    do
 *    {
 *      while(chunk) { foo(); i++ }
 *      chunk = GOMP_loop_ordered_runtime_next();
 *    } while(chunk);
 *    GOMP_loop_end(); 
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_runtime_start,
              PROTOTYPE(long start, long end, long incr, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(start, end, incr, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("start=%ld end=%ld incr=%ld chunk_size=%ld istart=%p iend=%p", start, end, incr, chunk_size, istart, iend));

/**
 * GOMP_loop_static_next
 * Avail: GCC >= 4.2
 * 
 * All threads call this to schedule the next chunk of work. This wrapper only triggers 
 * after GOMP_loop_doacross_static_start. Should also trigger after GOMP_loop_static_start, 
 * but the code transformation of the compiler never seems to add such call. Since 
 * GOMP_loop_doacross_static_start is not available in older versions of the
 * compiler, we actually see this wrapper being called starting at GCC >= 6.
 */
XTR_WRAP_GOMP(GOMP_loop_static_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_dynamic_next
 * Avail: GCC >= 4.2
 *
 * All threads call this to schedule the next chunk of work. We see this 
 * in all kinds of loop constructs:  GOMP_loop_doacross_dynamic_start 
 * (#pragma omp for ordered, GCC >= 6), GOMP_loop_dynamic_start (#pragma omp for), 
 * and GOMP_parallel_loop_dynamic (#pragma omp parallel for).
 */
XTR_WRAP_GOMP(GOMP_loop_dynamic_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_guided_next
 * Avail: GCC >= 4.2
 * 
 * All threads call this to schedule the next chunk of work. We see this 
 * in all kinds of loop constructs:  GOMP_loop_doacross_guided_start 
 * (#pragma omp for ordered, GCC >= 6), GOMP_loop_guided_start (#pragma omp for), 
 * and GOMP_parallel_loop_guided (#pragma omp parallel for).
 */
XTR_WRAP_GOMP(GOMP_loop_guided_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_runtime_next
 * Avail: GCC >= 4.2
 * 
 * All threads call this to schedule the next chunk of work. We see this 
 * in all kinds of loop constructs:  GOMP_loop_doacross_runtime_start 
 * (#pragma omp for ordered, GCC >= 6), GOMP_loop_runtime_start (#pragma omp for), 
 * and GOMP_parallel_loop_runtime (#pragma omp parallel for).
 */
XTR_WRAP_GOMP(GOMP_loop_runtime_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_nonmonotonic_dynamic_next
 * Avail: GCC >= 6
 * 
 * Replaces GOMP_loop_dynamic_next starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_nonmonotonic_dynamic_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_nonmonotonic_guided_next
 * Avail: GCC >= 6
 * 
 * Replaces GOMP_loop_guided_next starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_nonmonotonic_guided_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_nonmonotonic_runtime_next
 * Avail: GCC >= 9
 * 
 * Replaces GOMP_loop_runtime_next starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_nonmonotonic_runtime_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_maybe_nonmonotonic_runtime_next
 * Avail: GCC >= 9
 * 
 * Replaces GOMP_loop_runtime_next starting at GCC >= 9
 */
XTR_WRAP_GOMP(GOMP_loop_maybe_nonmonotonic_runtime_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_ordered_static_next
 * Avail: GCC >= 4.2
 *
 * Replaces GOMP_loop_static_next when the loop construct includes an 'ordered' clause.
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_static_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_ordered_dynamic_next
 * Avail: GCC >= 4.2
 * 
 * Replaces GOMP_loop_dynamic_next when the loop construct includes an 'ordered' clause.
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_dynamic_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_ordered_guided_next
 * Avail: GCC >= 4.2
 * 
 * Replaces GOMP_loop_guided_next when the loop construct includes an 'ordered' clause.
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_guided_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_ordered_runtime_next
 * Avail: GCC >= 4.2
 * 
 * Replaces GOMP_loop_runtime_next when the loop construct includes an 'ordered' clause.
 */
XTR_WRAP_GOMP(GOMP_loop_ordered_runtime_next,
              PROTOTYPE(long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("istart=%p iend=%p", istart, iend));

/**
 * GOMP_loop_doacross_static_start
 * Avail: GCC >= 6
 *
 * Wrapper generated for ordered loops with cross-iteration dependences that allow doacross parallelism.
 * Work chunks are obtained through GOMP_loop_doacross_static_start and GOMP_loop_static_next, as usual,
 * but the runtime triggers extra POST/WAIT operations with GOMP_doacross_post and GOMP_doacross_wait to
 * signal completion of current iteration and wait for completion of dependant iteration.
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered(1) schedule(static, 1)
 *    for (i=1; i<N; i++)
 *    {
 *      A[i] = foo(i);
 *      #pragma omp ordered depend(sink: i-1)
 *      B[i] = bar(A[i], B[i-1]);
 *      #pragma omp ordered depend(source)
 *      C[i] = baz(B[i]);
 *    }
 * >>>
 *    GOMP_loop_doacross_static_start();
 *    loop
 *    {
 *      GOMP_doacross_post();
 *      GOMP_doacross_wait();
 *      GOMP_doacross_post();
 *      GOMP_doacross_wait();
 *      ..
 *      GOMP_doacross_post();
 *    }
 *    GOMP_loop_static_next();
 *    GOMP_loop_end();
 */
XTR_WRAP_GOMP(GOMP_loop_doacross_static_start,
              PROTOTYPE(unsigned ncounts, long *counts, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(ncounts),
              REAL_SYMBOL_ARGS(ncounts, counts, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("ncounts=%u counts=%p chunk_size=%ld istart=%p iend=%p", ncounts, counts, chunk_size, istart, iend));

/**
 * GOMP_loop_doacross_dynamic_start
 * Avail: GCC >= 6
 * 
 * Wrapper generated for ordered loops with cross-iteration dependences that allow doacross parallelism.
 * Work chunks are obtained through GOMP_loop_doacross_dynamic_start and GOMP_loop_dynamic_next, as usual, 
 * but the runtime triggers extra POST/WAIT operations with GOMP_doacross_post and GOMP_doacross_wait to 
 * signal completion of current iteration and wait for completion of dependant iteration.
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered(1) schedule(dynamic, 1)
 *    for (i=1; i<N; i++)
 *    {
 *      A[i] = foo(i);
 *      #pragma omp ordered depend(sink: i-1)
 *      B[i] = bar(A[i], B[i-1]);
 *      #pragma omp ordered depend(source)
 *      C[i] = baz(B[i]);
 *    }
 * >>>
 *    GOMP_loop_doacross_dynamic_start();
 *    loop
 *    {
 *      GOMP_doacross_post();
 *      GOMP_loop_dynamic_next();
 *      GOMP_doacross_wait();
 *      GOMP_doacross_post();
 *      GOMP_loop_dynamic_next();
 *      GOMP_doacross_wait();
 *      ...
 *      GOMP_doacross_post();
 *      GOMP_loop_dynamic_next();
 *    }
 *    GOMP_loop_end();
 */
XTR_WRAP_GOMP(GOMP_loop_doacross_dynamic_start,
              PROTOTYPE(unsigned ncounts, long *counts, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(ncounts),
              REAL_SYMBOL_ARGS(ncounts, counts, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("ncounts=%u counts=%p chunk_size=%ld istart=%p iend=%p", ncounts, counts, chunk_size, istart, iend));

/**
 * GOMP_loop_doacross_guided_start
 * Avail: GCC >= 6
 * 
 * Wrapper generated for ordered loops with cross-iteration dependences that allow doacross parallelism.
 * Work chunks are obtained through GOMP_loop_doacross_guided_start and GOMP_loop_guided_next, as usual,
 * but the runtime triggers extra POST/WAIT operations with GOMP_doacross_post and GOMP_doacross_wait to
 * signal completion of current iteration and wait for completion of dependant iteration.
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered(1) schedule(guided, 1)
 *    for (i=1; i<N; i++)
 *    {
 *      A[i] = foo(i);
 *      #pragma omp ordered depend(sink: i-1)
 *      B[i] = bar(A[i], B[i-1]);
 *      #pragma omp ordered depend(source)
 *      C[i] = baz(B[i]);
 *    }
 * >>>
 *    GOMP_loop_doacross_guided_start();
 *    loop
 *    {
 *      GOMP_doacross_post();
 *      GOMP_doacross_wait();
 *      GOMP_doacross_post();
 *      GOMP_doacross_wait();
 *      ..
 *      GOMP_doacross_post();
 *    }
 *    GOMP_loop_guided_next();
 *    GOMP_loop_end();
 */
XTR_WRAP_GOMP(GOMP_loop_doacross_guided_start,
              PROTOTYPE(unsigned ncounts, long *counts, long chunk_size, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(ncounts),
              REAL_SYMBOL_ARGS(ncounts, counts, chunk_size, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("ncounts=%u counts=%p chunk_size=%ld istart=%p iend=%p", ncounts, counts, chunk_size, istart, iend));

/**
 * GOMP_loop_doacross_runtime_start
 * Avail: GCC >= 6
 * 
 * Wrapper generated for ordered loops with cross-iteration dependences that allow doacross parallelism.
 * Work chunks are obtained through GOMP_loop_doacross_runtime_start and GOMP_loop_runtime_next, as usual,
 * but the runtime triggers extra POST/WAIT operations with GOMP_doacross_post and GOMP_doacross_wait to
 * signal completion of current iteration and wait for completion of dependant iteration.
 *
 * Code transformation:
 * <<<
 *    #pragma omp for ordered(1) schedule(runtime, 1)
 *    for (i=1; i<N; i++)
 *    {
 *      A[i] = foo(i);
 *      #pragma omp ordered depend(sink: i-1)
 *      B[i] = bar(A[i], B[i-1]);
 *      #pragma omp ordered depend(source)
 *      C[i] = baz(B[i]);
 *    }
 * >>>
 *    GOMP_loop_doacross_runtime_start();
 *    loop
 *    {
 *      GOMP_doacross_post();
 *      GOMP_loop_runtime_next();
 *      GOMP_doacross_wait();
 *      GOMP_doacross_post();
 *      GOMP_loop_runtime_next();
 *      GOMP_doacross_wait();
 *      ...
 *      GOMP_doacross_post();
 *      GOMP_loop_runtime_next();
 *    }
 *    GOMP_loop_end();
 */
XTR_WRAP_GOMP(GOMP_loop_doacross_runtime_start,
              PROTOTYPE(unsigned ncounts, long *counts, long *istart, long *iend),
              RETURN(int),
              ENTRY_PROBE_ARGS(ncounts),
              REAL_SYMBOL_ARGS(ncounts, counts, istart, iend),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("ncounts=%u counts=%p istart=%p iend=%p", ncounts, counts, istart, iend));

/**
 * GOMP_parallel_loop_static_start
 * Avail: GCC >= 4.2 (USED ONLY IN GCC <4.9)
 *
 * Wrapper generated for parallel loops with schedule(static). In GCC 4, this is
 * split into a non-blocking GOMP_parallel_loop_static_start(), and a blocking 
 * GOMP_parallel_end() call. If we have schedule(static) and no ordered, then we 
 * ought to be able to get away with no worksharing context at all, since the 
 * runtime can simply perform the arithmetic directly in each thread to divide
 * up the iterations. Which would mean that there will no GOMP_*_next() nor 
 * GOMP_loop_end*() calls involved. In this case, the compiler seems to replace 
 * GOMP_parallel_loop_static_start() by the more general GOMP_parallel_start().
 *
 * A very significant difference in handling parallel regions between GCC 4 and newer 
 * versions is the way the outlined function is invoked. In GCC 4, the master
 * thread opening parallelism, and the new spawned threads for this parallel,
 * follow distinct paths to invoke the outlined function. The new threads are 
 * passed the function pointer to the outlined for them to jump. But for the
 * master, the compiler transformation adds a direct call to the outlined right 
 * after GOMP_parallel_loop_static_start() returns from creating threads. In order 
 * to mark the execution of the outlined properly, the emission of events needs to
 * happen from two points: 1) the outlined function pointer passed to the
 * threads is replaced by an instrumented trampoline that marks begin and end, and
 * 2) as the master thread's direct call to the outlined can not be swapped into
 * the instrumented trampoline, the exit probe of this wrapper marks the start of
 * the master's outlined function until GOMP_parallel_end is entered.
 *
 * In GCC >= 4.9, both the _start() and _end() routines are replaced by a single 
 * blocking call (see GOMP_parallel_loop_static). Furthermore, all threads receive 
 * the function pointer to the outlined, so replacing the outlined by our instrumented 
 * trampoline captures the activiy of all threads alike, including the master. 
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(static, 1)
 *    for (i=0, i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_start(main._omp_fn.0, ...) // Spawned threads will call the outlined through this pointer (swap into an instrumented trampoline)
 *    main._omp_fn.0(...) // Master thread invokes the outlined directly right after GOMP_parallel_start
 *    GOMP_parallel_end()
 */
XTR_WRAP_GOMP_FORK_START( GOMP_parallel_loop_static_start,
                          PROTOTYPE(void (*fn)(void *),
                                    void *data,
                                    unsigned num_threads,
                                    long start, 
                                    long end, 
                                    long incr, 
                                    long chunk_size),
                          NO_RETURN,
                          ENTRY_PROBE_ARGS(),
                          REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size),
                          EXIT_PROBE_ARGS(fn),
                          DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld", 
                                    fn, data, num_threads, start, end, incr, chunk_size));

/**
 * GOMP_parallel_loop_dynamic_start
 * Avail: GCC >= 4.2 (USED ONLY IN GCC 4)
 *
 * Wrapper generated for parallel loops with schedule(dynamic). In GCC 4, this is
 * split into a non-blocking GOMP_parallel_loop_dynamic_start(), and a blocking
 * GOMP_parallel_end() call. Between these two, a worksharing context is created
 * where calls to GOMP_loop_dynamic_next() are used to fetch new work chunks, 
 * and the loop worksharing is ended with GOMP_loop_end_nowait(). 
 *
 * A very significant difference in handling parallel regions between GCC 4 and newer
 * versions is the way the outlined function is invoked. In GCC 4, the master
 * thread opening parallelism, and the new spawned threads for this parallel,
 * follow distinct paths to invoke the outlined function. The new threads are
 * passed the function pointer to the outlined for them to jump. But for the
 * master, the compiler transformation adds a direct call to the outlined right
 * after GOMP_parallel_loop_dynamic_start() returns from creating threads. In order
 * to mark the execution of the outlined properly, the emission of events needs to
 * happen from two points: 1) the outlined function pointer passed to the
 * threads is replaced by an instrumented trampoline that marks begin and end, and
 * 2) as the master thread's direct call to the outlined can not be swapped into
 * the instrumented trampoline, the exit probe of this wrapper marks the start of
 * the master's outlined function until GOMP_parallel_end is entered.
 *
 * In GCC >= 4.9, both the _start() and _end() routines are replaced by a single
 * blocking call (see GOMP_parallel_loop_dynamic). Furthermore, all threads receive the 
 * function pointer to the outlined, so replacing the outlined by our instrumented 
 * trampoline captures the activiy of all threads alike, including the master.
 * 
 * In GCC >= 9, *loop_dynamic* calls may change into *loop_nonmonotonic_dynamic*.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(dynamic)
 *    for (i=0, i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_loop_dynamic_start(main._omp_fn.0, ...) // Spawned threads will call the outlined through this pointer (swap into an instrumented trampoline)
 *    main._omp_fn.0(...) // Master thread invokes the outlined directly right after GOMP_parallel_loop_dynamic_start
 *    GOMP_parallel_end()
 *
 *    main.omp_fn.0(...)
 *    {
 *      while ((chunk = GOMP_loop_dynamic_next()))
 *      {
 *        foo(); i++;
 *      }
 *      GOMP_loop_end_nowait();
 *    }
 */
XTR_WRAP_GOMP_FORK_START( GOMP_parallel_loop_dynamic_start,
                          PROTOTYPE(void (*fn)(void *),
                                    void *data,
                                    unsigned num_threads,
                                    long start, 
                                    long end, 
                                    long incr, 
                                    long chunk_size),
                          NO_RETURN,
                          ENTRY_PROBE_ARGS(),
                          REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size),
                          EXIT_PROBE_ARGS(fn),
                          DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld", 
                                    fn, data, num_threads, start, end, incr, chunk_size));

/**
 * GOMP_parallel_loop_guided_start
 * Avail: GCC >= 4.2 (USED ONLY IN GCC 4)
 *
 * Wrapper generated for parallel loops with schedule(guided). In GCC 4, this is
 * split into a non-blocking GOMP_parallel_loop_guided_start(), and a blocking
 * GOMP_parallel_end() call. Between these two, a worksharing context is created
 * where calls to GOMP_loop_guided_next() are used to fetch new work chunks, 
 * and the loop worksharing is ended with GOMP_loop_end_nowait(). 
 *
 * A very significant difference in handling parallel regions between GCC 4 and newer
 * versions is the way the outlined function is invoked. In GCC 4, the master
 * thread opening parallelism, and the new spawned threads for this parallel,
 * follow distinct paths to invoke the outlined function. The new threads are
 * passed the function pointer to the outlined for them to jump. But for the
 * master, the compiler transformation adds a direct call to the outlined right
 * after GOMP_parallel_loop_guided_start() returns from creating threads. In order
 * to mark the execution of the outlined properly, the emission of events needs to
 * happen from two points: 1) the outlined function pointer passed to the
 * threads is replaced by an instrumented trampoline that marks begin and end, and
 * 2) as the master thread's direct call to the outlined can not be swapped into
 * the instrumented trampoline, the exit probe of this wrapper marks the start of
 * the master's outlined function until GOMP_parallel_end is entered.
 * 
 * In GCC >= 4.9, both the _start() and _end() routines are replaced by a single 
 * blocking call (see GOMP_parallel_loop_guided). Furthermore, all threads receive the
 * function pointer to the outlined, so replacing the outlined by our instrumented
 * trampoline captures the activiy of all threads alike, including the master.
 *
 * In GCC >= 9, *loop_guided* calls may change into *loop_nonmonotonic_guided*.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(guided)
 *    for (i=0, i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_loop_guided_start(main._omp_fn.0, ...) // Spawned threads will call the outlined through this pointer (swap into an instrumented trampoline)
 *    main._omp_fn.0(...) // Master thread invokes the outlined directly right after GOMP_parallel_loop_guided_start
 *    GOMP_parallel_end()
 *
 *    main.omp_fn.0(...)
 *    {
 *      while ((chunk = GOMP_loop_guided_next()))
 *      {
 *        foo(); i++;
 *      }
 *      GOMP_loop_end_nowait();
 *    }
 */
XTR_WRAP_GOMP_FORK_START( GOMP_parallel_loop_guided_start,
                          PROTOTYPE(void (*fn)(void *),
                                    void *data,
                                    unsigned num_threads,
                                    long start, 
                                    long end, 
                                    long incr, 
                                    long chunk_size),
                          NO_RETURN,
                          ENTRY_PROBE_ARGS(),
                          REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size),
                          EXIT_PROBE_ARGS(fn),
                          DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld", 
                                    fn, data, num_threads, start, end, incr, chunk_size));

/**
 * GOMP_parallel_loop_runtime_start
 * Avail: GCC >= 4.2 (USED ONLY IN GCC 4)
 *
 * Wrapper generated for parallel loops with schedule(runtime). In GCC 4, this is
 * split into a non-blocking GOMP_parallel_loop_runtime_start(), and a blocking
 * GOMP_parallel_end() call. Between these two, a worksharing context is created
 * where calls to GOMP_loop_runtime_next() are used to fetch new work chunks, 
 * and the loop worksharing is ended with GOMP_loop_end_nowait(). 
 *
 * A very significant difference in handling parallel regions between GCC 4 and newer
 * versions is the way the outlined function is invoked. In GCC 4, the master
 * thread opening parallelism, and the new spawned threads for this parallel,
 * follow distinct paths to invoke the outlined function. The new threads are
 * passed the function pointer to the outlined for them to jump. But for the
 * master, the compiler transformation adds a direct call to the outlined right
 * after GOMP_parallel_loop_runtime_start() returns from creating threads. In order
 * to mark the execution of the outlined properly, the emission of events needs to
 * happen from two points: 1) the outlined function pointer passed to the
 * threads is replaced by an instrumented trampoline that marks begin and end, and
 * 2) as the master thread's direct call to the outlined can not be swapped into
 * the instrumented trampoline, the exit probe of this wrapper marks the start of
 * the master's outlined function until GOMP_parallel_end is entered.
 *
 * In GCC >= 4.9, both the _start() and _end() routines are replaced by a single 
 * blocking call (see GOMP_parallel_loop_runtime). Furthermore, all threads receive the
 * function pointer to the outlined, so replacing the outlined by our instrumented
 * trampoline captures the activiy of all threads alike, including the master.
 *
 * In GCC >= 9, *loop_runtime* calls may change into *loop_maybe_nonmonotonic_runtime*.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(runtime)
 *    for (i=0, i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_loop_runtime_start(main._omp_fn.0, ...) // Spawned threads will call the outlined through this pointer (swap into an instrumented trampoline)
 *    main._omp_fn.0(...) // Master thread invokes the outlined directly right after GOMP_parallel_loop_runtime_start
 *    GOMP_parallel_end()
 *
 *    main.omp_fn.0(...)
 *    {
 *      while ((chunk = GOMP_loop_runtime_next()))
 *      {
 *        foo(); i++;
 *      }
 *      GOMP_loop_end_nowait();
 *    }
 */
XTR_WRAP_GOMP_FORK_START(GOMP_parallel_loop_runtime_start,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr),
                   EXIT_PROBE_ARGS(fn),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld", 
                             fn, data, num_threads, start, end, incr));

/**
 * GOMP_parallel_loop_static
 * Avail: GCC >= 4.9
 *
 * Replaces the pair of non-blocking GOMP_parallel_loop_static_start() + blocking GOMP_parallel_end()
 * used by older versions of the compiler, with a single blocking call. All threads (including the master)
 * are passed a pointer to the outlined function for them to jump, that we swap into an instrumented
 * trampoline to capture the begin and end of the outlined execution. If we have schedule(static) and no 
 * ordered, then we ought to be able to get away with no worksharing context at all, since the
 * runtime can simply perform the arithmetic directly in each thread to divide up the iterations. 
 * Which would mean that there will no GOMP_*_next() nor GOMP_loop_end*() calls involved. In this case, 
 * the compiler seems to replace GOMP_parallel_loop_static() by the more general GOMP_parallel().
 * See GOMP_parallel_loop_static_start for details.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(static)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel(main._omp_fn.0, ...)
 *
 *    main._omp_fn.0(...)
 *    {
 *      * compute start and end iterations statically 
 *      for (i=start; i<end; i++)
 *      {
 *        foo();
 *      }
 *    }
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_static,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             long chunk_size, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, chunk_size, flags));

/**
 * GOMP_parallel_loop_dynamic
 * Avail: GCC >= 4.9
 *
 * Replaces the pair of non-blocking GOMP_parallel_loop_dynamic_start() + blocking GOMP_parallel_end()
 * used by older versions of the compiler, with a single blocking call. All threads (including the master)
 * are passed a pointer to the outlined function for them to jump, that we swap into an instrumented
 * trampoline to capture the begin and end of the outlined execution. See GOMP_parallel_loop_dynamic_start for details.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(dynamic)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_loop_dynamic(main.omp_fn.0, ...)
 *
 *    main.omp_fn.0(...)
 *    {
 *      while ((chunk = GOMP_loop_dynamic_next()))
 *      {
 *        foo(); i++;
 *      }
 *      GOMP_loop_end_nowait();
 *    }
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_dynamic,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             long chunk_size, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, chunk_size, flags));

/**
 * GOMP_parallel_loop_guided
 * Avail: GCC >= 4.9
 *
 * Replaces the pair of non-blocking GOMP_parallel_loop_guided_start() + blocking GOMP_parallel_end()
 * used by older versions of the compiler, with a single blocking call. All threads (including the master)
 * are passed a pointer to the outlined function for them to jump, that we swap into an instrumented
 * trampoline to capture the begin and end of the outlined execution. See GOMP_parallel_loop_guided_start for details.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(guided)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_loop_guided(main.omp_fn.0, ...)
 *
 *    main.omp_fn.0(...)
 *    {
 *      while ((chunk = GOMP_loop_guided_next()))
 *      {
 *        foo(); i++;
 *      }
 *      GOMP_loop_end_nowait();
 *    }
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_guided,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             long chunk_size, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, chunk_size, flags));

/**
 * GOMP_parallel_loop_runtime
 * Avail: GCC >= 4.9
 *
 * Replaces the pair of non-blocking GOMP_parallel_loop_runtime_start() + blocking GOMP_parallel_end()
 * used by older versions of the compiler, with a single blocking call. All threads (including the master)
 * are passed a pointer to the outlined function for them to jump, that we swap into an instrumented
 * trampoline to capture the begin and end of the outlined execution. See GOMP_parallel_loop_runtime_start for details.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel for schedule(runtime)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_loop_runtime(main.omp_fn.0, ...)
 *
 *    main.omp_fn.0(...)
 *    {
 *      while ((chunk = GOMP_loop_runtime_next()))
 *      {
 *        foo(); i++;
 *      }
 *      GOMP_loop_end_nowait();
 *    }
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_runtime,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, flags));

/**
 * GOMP_parallel_loop_nonmonotonic_dynamic
 * Avail: GCC >= 6
 *
 * Replaces GOMP_parallel_loop_dynamic starting from GCC >= 9. 
 * Calls to GOMP_loop_dynamic_next() in the inner worksharing context also change into 
 * GOMP_loop_nonmonotonic_dynamic_next(). See GOMP_parallel_loop_dynamic for details.
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_nonmonotonic_dynamic,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             long chunk_size, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, chunk_size, flags));

/**
 * GOMP_parallel_loop_nonmonotonic_guided
 * Avail: GCC >= 6
 *
 * Replaces GOMP_parallel_loop_guided starting from GCC >= 9.
 * Calls to GOMP_loop_guided_next() in the inner worksharing context also change into
 * GOMP_loop_nonmonotonic_guided_next(). See GOMP_parallel_loop_guided for details.
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_nonmonotonic_guided,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             long chunk_size, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, chunk_size, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld chunk_size=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, chunk_size, flags));

/**
 * GOMP_parallel_loop_nonmonotonic_runtime
 * Avail: GCC >= 6
 *
 * Replaces GOMP_parallel_loop_runtime starting from GCC >= 9.
 * Calls to GOMP_loop_runtime_next() in the inner worksharing context also change into
 * GOMP_loop_nonmonotonic_runtime_next(). See GOMP_parallel_loop_runtime for details.
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_nonmonotonic_runtime,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, flags));

/**
 * GOMP_parallel_loop_maybe_nonmonotonic_runtime
 * Avail: GCC >= 6
 *
 * Replaces GOMP_parallel_loop_runtime starting from GCC >= 9.
 * Calls to GOMP_loop_runtime_next() in the inner worksharing context also change into
 * GOMP_loop_maybe_nonmonotonic_runtime_next(). See GOMP_parallel_loop_runtime for details.
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_loop_maybe_nonmonotonic_runtime,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             long start, 
                             long end, 
                             long incr, 
                             unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, start, end, incr, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u start=%ld end=%ld incr=%ld flags=%u", 
                             fn, data, num_threads, start, end, incr, flags));

/**
 * GOMP_loop_end
 * Avail: GCC >= 4.2
 * 
 * This ends loop constructs with an implicit barrier (join) in all threads, i.e.:
 * GOMP_loop_doacross_*_start and GOMP_loop_ordered_*_start (#pragma omp for ordered), 
 * GOMP_loop_*_start (#pragma omp for), and GOMP_parallel_loop_*_start (#pragma omp parallel for).
 *
 * Triggers after one last call to GOMP_loop_*_next where there aren't more chunks of work. 
 */
XTR_WRAP_GOMP(GOMP_loop_end,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_loop_end_nowait
 * Avail: GCC >= 4.2
 *
 * This ends loop constructs without implicit barrier in all threads. 
 * Triggers usually for parallel loops and when the 'nowait' clause is set,
 * after one last call to GOMP_loop_*_next where there aren't more chunks of work.
 */
XTR_WRAP_GOMP(GOMP_loop_end_nowait,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/*********************/
/* libgomp/ordered.c */
/*********************/

/**
 * GOMP_ordered_start
 * Avail: GCC >= 4.2
 *
 * Triggers when encountering the start of an ordered block.
 * If the current thread is not at the head of the queue, it blocks.
 *
 * Code transformation:
 * <<<
 *    #pragma omp ordered
 *    foo()
 * >>>
 *    GOMP_ordered_start()
 *    foo()
 *    GOMP_ordered_end()
 */
XTR_WRAP_GOMP(GOMP_ordered_start,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_ordered_end
 * Avail: GCC >= 4.2
 *
 * Triggers when reaching the end of an ordered block.
 * See GOMP_ordered_start for details.
 */
XTR_WRAP_GOMP(GOMP_ordered_end,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_doacross_post
 * Avail: GCC >= 6
 *
 * Signals completion of iteration i in doacross loops.
 * See GOMP_loop_doacross_*_start for details.
 */
XTR_WRAP_GOMP(GOMP_doacross_post,
              PROTOTYPE(long *counts),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(counts),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("counts=%p", counts));

/**
 * GOMP_doacross_wait_varargs
 *
 * Intermediate wrapper to the real GOMP_doacross_wait symbol. 
 * GOMP_doacross_wait prototype receives a variable argument list '...'
 * that can not be propagated in the call to the real symbol unless 
 * converted into a 'va_list' that is received here in 'varargs', 
 * and then unpacked to call the real symbol 'real_sym_ptr' 
 * with the correct number of arguments. Calling the real GOMP_doacross_wait
 * needs a switch with as many cases as possible values of 'ncounts'. 
 * These cases are generated dynamically with the script
 * 'genstubs-libgomp.sh' up to a maximum of DOACROSS_MAX_ARGS, 
 * defined in that script. See XTR_OMP_WRAP_VARARGS for details.
 *
 * @param real_sym_ptr Function pointer to the real implementation of GOMP_doacros_wait
 * @param first        Original argument received in GOMP_doacross_wait wrapper
 * @param va_list      Packs the variable arguments '...' received in GOMP_doacross_wait wrapper
 */
static void GOMP_doacross_wait_varargs(void (*REAL_SYMBOL_PTR(GOMP_doacross_wait))(long, ...), long first, va_list varargs)
{
	unsigned i = 0;
	long args[MAX_DOACROSS_ARGS];

	// Retrieve ncounts stored at GOMP_loop_doacross_*_start from the thread TLS 
	unsigned ncounts = __GOMP_retrieve_doacross_ncounts();

  // Fetch ncounts number of arguments from varargs
  for (i = 0; i < ncounts; i++)
  {
    args[i] = va_arg(varargs, long);
  }

  // Call GOMP_doacross_wait with the proper number of arguments
  switch (ncounts)
  {
    #include "gnu-libgomp-intermediate/libgomp-doacross-intermediate-switch.c"
    default:
      THREAD_ERROR("Unhandled GOMP_doacross_wait call with %d arguments! "
			             "Re-run the script 'genstubs-libgomp.sh' increasing the value of DOACROSS_MAX_ARGS and recompile Extrae. "
			             "Quitting!\n", ncounts);
      exit(-1);
      break;
  }
}

/**
 * GOMP_doacross_wait
 * Avail: GCC >= 6
 *
 * Waits for completion of iteration i-1 in doacross loops
 * See GOMP_loop_doacross_*_start for details.
 */
XTR_WRAP_GOMP(GOMP_doacross_wait,
              PROTOTYPE(long first, ...),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(first, varargs),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("first=%ld", first), 
              VARARGS(first));

/**********************/
/* libgomp/parallel.c */
/**********************/

/**
 * GOMP_parallel_start
 * Avail: GCC >= 4.2 (ONLY USED IN GCC 4)
 *
 * Wrapper enclosing a parallel region. In GCC 4, these are split into a non-blocking 
 * GOMP_parallel_start() that spawns threads, and a blocking GOMP_parallel_end() call.
 * 
 * A very significant difference in handling parallel regions between GCC 4 and newer
 * versions is the way the outlined function is invoked. In GCC 4, the master
 * thread opening parallelism, and the new spawned threads for this parallel,
 * follow distinct paths to invoke the outlined function. The new threads are
 * passed the function pointer to the outlined for them to jump. But for the
 * master, the compiler transformation adds a direct call to the outlined right
 * after GOMP_parallel_start() returns from creating threads. In order
 * to mark the execution of the outlined properly, the emission of events needs to
 * happen from two points: 1) the outlined function pointer passed to the
 * threads is replaced by an instrumented trampoline that marks begin and end, and
 * 2) as the master thread's direct call to the outlined can not be swapped into
 * the instrumented trampoline, the exit probe of this wrapper marks the start of
 * the master's outlined function until GOMP_parallel_end is entered.
 *
 * In GCC >= 4.9, both the _start() and _end() routines are replaced by a single
 * blocking GOMP_parallel(). Furthermore, all threads receive the function pointer 
 * to the outlined, so replacing the outlined by our instrumented trampoline 
 * captures the activiy of all threads alike, including the master.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel_start(main._omp_fn.0, ...) // Spawned threads will call the outlined through this pointer (swap into an instrumented trampoline)
 *    main._omp_fn.0(...); // Master thread invokes the outlined directly right after GOMP_parallel_start
 *    GOMP_parallel_end()
 * 
 *    main._omp_fn.0(...)
 *    {
 *      foo();
 *    }
 */
XTR_WRAP_GOMP_FORK_START(GOMP_parallel_start,
                         PROTOTYPE(void (*fn)(void *), void *data, unsigned num_threads),
                         NO_RETURN,
                         ENTRY_PROBE_ARGS(),
                         REAL_SYMBOL_ARGS(fn, data, num_threads),
                         EXIT_PROBE_ARGS(fn),
                         DEBUG_ARGS("fn=%p data=%p num_threads=%u", fn, data, num_threads));

/**
 * GOMP_parallel_end
 * Avail: GCC >= 4.2 (ONLY USED IN GCC 4)
 *
 * Wrapper marking the end of a parallel section. See GOMP_parallel_start for details.
 */
XTR_WRAP_GOMP_FORK_END( GOMP_parallel_end,
                         PROTOTYPE(),
                         NO_RETURN,
                         ENTRY_PROBE_ARGS(),
                         REAL_SYMBOL_ARGS(),
                         EXIT_PROBE_ARGS(),
                         DEBUG_ARGS(""));
                         
/**
 * GOMP_parallel
 * Avail: GCC >= 4.9
 *
 * Replaces the pair of non-blocking GOMP_parallel_start() + blocking GOMP_parallel_end()
 * used by older versions of the compiler, with a single blocking call. All threads 
 * (including the master) are passed a pointer to the outlined function for them to jump, 
 * that we swap into an instrumented trampoline to capture the begin and end of the 
 * outlined execution. See GOMP_parallel_start for details.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_parallel(main._omp_fn.0, ...)
 *    
 *    main._omp_fn.0(...)
 *    {
 *      foo();
 *    }
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             unsigned int flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u flags=%u", fn, data, num_threads, flags));

/******************/
/* libgomp/task.c */
/******************/

/**
 * GOMP_task
 * Avail: GCC >= 4.4 but increased the parameters in 4.9 and later in 6.0 and
 * 9.0, to add new parameters 'depend', 'priority_arg' and 'detach',
 * respectively.
 *
 * Wrapper for #pragma omp task. As with parallel regions, the compiler transforms
 * the task body into an outlined function, that is passed by function pointer to 
 * the thread assigned to run the task. We swap this pointer into an instrumented 
 * trampoline to capture the execution of the task. 
 *
 * We define this prototype as receiving varargs, and we check the runtime version 
 * to decide with how many parameters we will make the call to the real function.
 *
 * The arguments holding the function's address, data, and copy function must be 
 * named as follows: 'fn' for the function's address, 'data' for the data, 
 * and 'copyfn' for the copy function. These specific names are accessed 
 * within GOMP_TASK_TRACING_LOGIC.
 * 
 * Code transformation:
 * <<<
 *    #pragma omp task
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_task(main._omp_fn.0, ...)
 *   
 *    main._omp_fn.0(...)
 *    {
 *      foo();
 *    }
 */
ATOMIC_COUNTER_INITIALIZER(__GOMP_task_counter, 0);

void GOMP_task_varargs(void (*REAL_SYMBOL_PTR(GOMP_task))(void *, void *, void *, long, long, int, unsigned, ...), 
                       void (*fn)(void *), 
											 void *data, 
											 void (*cpyfn)(void *, void *), 
											 long arg_size, 
											 long arg_align, 
											 int if_clause, 
											 unsigned flags, 
											 va_list varargs)
{
	void **depend       = NULL;
	int    priority_arg = 0;
	void  *detach       = NULL;

	if (__GOMP_version == GOMP_API_3_1)
	{
		REAL_SYMBOL_PTR(GOMP_task)(fn, data, cpyfn, arg_size, arg_align, if_clause, flags);
	}
	else if (__GOMP_version == GOMP_API_4_0)
	{
		depend = va_arg(varargs, void **);
		REAL_SYMBOL_PTR(GOMP_task)(fn, data, cpyfn, arg_size, arg_align, if_clause, flags, depend);
	}
	else if (__GOMP_version == GOMP_API_4_5)
	{
		depend = va_arg(varargs, void **);
		priority_arg = va_arg(varargs, int);
		REAL_SYMBOL_PTR(GOMP_task)(fn, data, cpyfn, arg_size, arg_align, if_clause, flags, depend, priority_arg);
	}
	else if (__GOMP_version >= GOMP_API_5_0)
	{
		depend = va_arg(varargs, void **);
		priority_arg = va_arg(varargs, int);
		detach  = va_arg(varargs, void *);
		if (__GOMP_version > GOMP_API_5_2)
		{
			THREAD_WARN("Call to GOMP_task v%f assuming varargs prototype hasn't changed from v%f (void **depend, int priority_arg, void *detach).",
			            __GOMP_version, GOMP_API_5_0);
			THREAD_WARN("Ensure the current prototype in libgomp/task.c matches and extend the GOMP_task wrapper to cover this case.");
		}
		REAL_SYMBOL_PTR(GOMP_task)(fn, data, cpyfn, arg_size, arg_align, if_clause, flags, depend, priority_arg, detach);
	}
}

XTR_WRAP_GOMP_TASK(GOMP_task,
                   PROTOTYPE(void (*fn)(void *), 
										         void *data, 
														 void (*cpyfn)(void *, void *), 
														 long arg_size, 
														 long arg_align, 
														 int if_clause, 
														 unsigned flags, 
														 ...),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, cpyfn, arg_size, arg_align, if_clause, flags, varargs),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p cpyfn=%p arg_size=%ld arg_align=%ld if_clause=%d flags=%u",
                              fn, data, cpyfn, arg_size, arg_align, if_clause, flags),
                   VARARGS(flags));

/**
 * GOMP_taskloop
 * Avail: GCC 6.0
 *
 * Wrapper for #pragma omp taskloop. As with tasks, the compiler transforms 
 * the taskloop body into an outlined function, that is passed by function 
 * pointer to the threads assigned to run the tasks. We swap this pointer
 * into an instrumented trampoline to capture the execution of the tasks.
 *
 * Code transformation:
 * <<< 
 *    #pragma omp taskloop num_tasks(20)
 *    for (i=0; i<NITERS; i++)
 *    {
 *			arr[i] = i * i;
 *    }
 * >>>
 *    GOMP_taskloop(main._omp_fn.0, data...)
 *
 *    main._omp_fn.0(data...)
 *    {
 *      for (i=data->start; i<data->end; i++)
 *      {
 *        arr[i] = i * i;
 *      }
 *    }
 */
XTR_WRAP_GOMP_TASKLOOP(GOMP_taskloop,
                   PROTOTYPE(void (*fn)(void *), 
										         void *data, 
														 void (*cpyfn)(void *, void *), 
														 long arg_size, 
														 long arg_align, 
														 unsigned flags,
                             unsigned long num_tasks, 
										         int priority, 
										         long start, 
										         long end, 
										         long step),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p cpyfn=%p arg_size=%ld arg_align=%ld flags=%u num_tasks=%lu priority=%d start=%ld end=%ld step=%ld",
		                           fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step));

XTR_WRAP_GOMP_TASKLOOP(GOMP_taskloop_ull,
                   PROTOTYPE(void (*fn)(void *), 
										         void *data, 
														 void (*cpyfn)(void *, void *), 
														 long arg_size, 
														 long arg_align, 
														 unsigned flags,
                             unsigned long num_tasks, 
										         int priority, 
										         unsigned long long start, 
										         unsigned long long end, 
										         unsigned long long step),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p cpyfn=%p arg_size=%ld arg_align=%ld flags=%u num_tasks=%lu priority=%d start=%llu end=%llu step=%llu",
		                           fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step));

/**
 * GOMP_taskwait
 * Avail: GCC >= 4.4
 *
 * Wrapper for the taskwait construct that waits on the completion of child tasks
 * of the current task.
 *
 * Code transformation:
 * <<<
 *    #pragma omp taskwait
 * >>>
 *    GOMP_taskwait();
 */
XTR_WRAP_GOMP(GOMP_taskwait,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_taskwait
 * Avail: GCC >= 4.7
 *
 * Wrapper for the taskyield construct has an empty implementaion in GNU libgomp as of 2025-02-28,
 * yet the compiler adds a call to GOMP_taskyield anyway.
 *
 * Code transformation:
 * <<<
 *    #pragma omp taskyield
 * >>>
 *    GOMP_taskyield();
 */
XTR_WRAP_GOMP(GOMP_taskyield,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_taskgroup_start
 * Avail: GCC >= 4.9
 *
 * Wrapper for the taskgroup construct that specifies a wait on completion of
 * tasks generated in the enclosing region until GOMP_taskgroup_end is reached.
 *
 * Code transformation:
 * <<<
 *    #pragma omp taskgroup
 *    {
 *       #pragma omp task
 *       {
 *         foo();
 *       }
 *    }
 * >>>
 *    GOMP_taskgroup_start();
 *    GOMP_task(main._omp_fn.0, ...);
 *    GOMP_taskgroup_end();
 *    
 *    main._omp_fn.0(...)
 *    {
 *      foo();
 *    }
 */
XTR_WRAP_GOMP(GOMP_taskgroup_start,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_taskgroup_end
 * Avail: GCC >= 4.9
 *
 * See GOMP_taskgroup_start for details.
 */
XTR_WRAP_GOMP(GOMP_taskgroup_end,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**********************/
/* libgomp/sections.c */
/**********************/

/**
 * GOMP_sections_start
 * Avail: GCC 4.2
 *
 * Triggers at the start of a sections construct, with the inner 'section' 
 * blocks compiled into a case statement. Subsequent blocks are retrieved by
 * GOMP_sections_next() until no more blocks are available and
 * GOMP_sections_end() is reached. 
 *
 * Code transformation:
 * <<<
 *    #pragma omp sections
 *    {
 *      #pragma omp section
 *      { foo(); }
 *      #pragma omp section
 *      { bar(); }
 *    }
 * >>>
 *    for (i = GOMP_sections_start(), i != 0; i = GOMP_sections_next())
 *    {
 *      switch(i)
 *      {
 *        case 1:
 *          foo(); break;
 *        case 2:
 *          bar(); break;
 *        default: 
 *          GOMP_sections_end();
 *      }
 */
XTR_WRAP_GOMP(GOMP_sections_start,
              PROTOTYPE(unsigned count),
              RETURN(unsigned),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(count),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("count=%u", count));
              
/**
 * GOMP_sections_next
 * Avail: GCC 4.2
 * 
 * See GOMP_sections_start for details.
 */
XTR_WRAP_GOMP(GOMP_sections_next,
              PROTOTYPE(),
              RETURN(unsigned),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/**
 * GOMP_parallel_sections_start
 * Avail: GCC >= 4.2 (USED ONLY IN GCC 4)
 *
 * Called by the master thread that opens a parallel sections block. Only in GCC 4, 
 * this is split into a non-blocking GOMP_parallel_sections_start() that spawns threads,
 * and a blocking GOMP_parallel_end(). Inner 'section' blocks are compiled into a case 
 * statement. Subsequent blocks are retrieved by GOMP_sections_next() until no more blocks 
 * are available and GOMP_sections_end_nowait() is reached.
 *
 * A very significant difference in handling parallel regions between GCC 4 and newer
 * versions is the way the outlined function is invoked. In GCC 4, the master
 * thread opening parallelism, and the new spawned threads for this parallel,
 * follow distinct paths to invoke the outlined function. The new threads are
 * passed the function pointer to the outlined for them to jump. But for the
 * master, the compiler transformation adds a direct call to the outlined right
 * after GOMP_parallel_sections_start() returns from creating threads. In order
 * to mark the execution of the outlined properly, the emission of events needs to
 * happen from two points: 1) the outlined function pointer passed to the
 * threads is replaced by an instrumented trampoline that marks begin and end, and
 * 2) as the master thread's direct call to the outlined can not be swapped into
 * the instrumented trampoline, the exit probe of this wrapper marks the start of
 * the master's outlined function until GOMP_parallel_end is entered.
 *
 * In GCC >= 4.9, both the _start() and _end() routines are replaced by a single
 * blocking call (see GOMP_parallel_sections). Furthermore, all threads receive the function 
 * pointer to the outlined, so replacing the outlined by our instrumented trampoline
 * captures the activiy of all threads alike, including the master.
 *
 * Code transformation:
 * <<<
 *    #pragma omp parallel sections
 *    {
 *      #pragma omp section
 *      { foo(); }       
 *      #pragma omp section
 *      { bar(); }
 *    }
 * >>>
 *    GOMP_parallel_sections_start(main._omp_fn.0, ...); // Spawned threads will call the outlined through this pointer (swap into an instrumented trampoline)
 *    main._omp_fn.0(...); // Master thread invokes the outlined directly right after GOMP_parallel_sections_start
 *    GOMP_parallel_end();
 * 
 *    main._omp_fn.0(...);
 *    {
 *      for (i = GOMP_sections_next(); i != 0; i = GOMP_sections_next())
 *      {
 *        switch(i)
 *        {
 *          case 1:
 *            foo(); break;
 *          case 2:
 *            bar(); break;
 *          default: 
 *            GOMP_sections_end_nowait();
 *        }
 *      }
 *    }
 */
XTR_WRAP_GOMP_FORK_START(GOMP_parallel_sections_start,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned num_threads,
                             unsigned count),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, count),
                   EXIT_PROBE_ARGS(fn),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u count=%u", 
                             fn, data, num_threads, count));

/**
 * GOMP_parallel_sections
 * Avail: GCC >= 4.9
 *
 * Replaces the pair of non-blocking GOMP_parallel_sections_start() + blocking GOMP_parallel_end()
 * used by older versions of the compiler, with a single blocking call. All threads (including the master)
 * are passed a pointer to the outlined function for them to jump, that we swap into an instrumented
 * trampoline to capture the begin and end of the outlined execution. See GOMP_parallel_sections_start for details.
 * 
 * Code transformation:
 * <<<
 *    #pragma omp parallel sections
 *    {
 *      #pragma omp section
 *      { foo(); }       
 *      #pragma omp section
 *      { bar(); }
 *    }
 * >>>
 *    GOMP_parallel_sections(main._omp_fn.0, ...);
 *
 *    main._omp_fn.0(...)
 *    {
 *      for (i = GOMP_sections_next(); i != 0; i = GOMP_sections_next())
 *      {
 *        switch(i)
 *        {
 *          case 1:
 *            foo(); break;
 *          case 2:
 *            bar(); break;
 *          default: 
 *            GOMP_sections_end_nowait();
 *        }
 *      }
 *    }
 */
XTR_WRAP_GOMP_FORK(GOMP_parallel_sections,
                   PROTOTYPE(void (*fn) (void *), void *data, unsigned num_threads, unsigned count, unsigned flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_threads, count, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_threads=%u count=%u flags=%u", fn, data, num_threads, count, flags));

/**
 * GOMP_sections_end
 * Avail: GCC >= 4.2
 *
 * Ends a sections block. See GOMP_sections_start for details.
 */
XTR_WRAP_GOMP(GOMP_sections_end,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));
              
/**
 * GOMP_sections_end_nowait
 * Avail: GCC >= 4.2
 * 
 * Ends a parallel sections block. See GOMP_parallel_sections_start / GOMP_parallel_sections for details.
 */
XTR_WRAP_GOMP(GOMP_sections_end_nowait,
              PROTOTYPE(),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/********************/
/* libgomp/single.c */
/********************/

/**
 * GOMP_single_start
 * Avail: GCC >= 4.2
 *
 * Initial scheduling for #pragma omp single. 
 * The compiler transformation only marks the beginning of the single section with a call to GOMP_single_start.
 * If there's no nowait clause, the compiler also adds a call to GOMP_barrier at the end. However, if nowait
 * is set, the compiler doesn't add anything else and there's no way to mark the end. Due to this, we can not
 * mark the SINGLE worksharing from the beginning to end, but only for the region that corresponds to the
 * initial scheduling. Usually, we mark the start regions as WORK SCHEDULING, but we don't in this particular
 * case because the events fall in the same timestamp as those that mark the SINGLE, and then the LastEvtVal
 * semantic never shows where the SINGLE happens.
 *
 * Code transformation:
 * <<<
 *    #pragma omp single
 *    {
 *      foo();
 *    }
 *.   bar ();
 * >>>
 *    switch(GOMP_single_start())
 *    {
 *      case 1:
 *      {
 *        foo();
 *        (GOMP_barrier();)
 *        break;
 *      }
 *      default:
 *        break;
 *    }
 *    bar();
 */
XTR_WRAP_GOMP(GOMP_single_start,
              PROTOTYPE(),
              RETURN(unsigned),
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS(""));

/*******************/ 
/* libgomp/teams.c */
/*******************/

/**
 * GOMP_teams_reg
 * Avail: GCC >= 9
 * 
 * Triggers with a #pragma omp teams region where the master thread of each team executes the teams region.
 * All threads (including the master) are passed a pointer to the outlined function for them to jump, 
 * that we swap into an instrumented trampoline to capture the begin and end of the outlined execution.
 * 
 * Code transformation:
 * <<<
 *    #pragma omp teams distribute parallel for num_teams(4) thread_limit(16)
 *    for (i=0; i<NITERS; i++)
 *    {
 *      foo();
 *    }
 * >>>
 *    GOMP_teams_reg(main._omp_fn.0, ...);
 * 
 *    main._omp_fn.0(...)
 *    {
 *      ...
 *      GOMP_parallel(main._omp_fn.1, ...);
 *      ...
 *    }
 *
 *    main._omp_fn.1(...)
 *    {
 *      for (i = start; i != end; i++)
 *      {
 *        foo();
 *      }
 *    }
 */
XTR_WRAP_GOMP_TEAMS(GOMP_teams_reg,
                   PROTOTYPE(void (*fn)(void *),
                             void *data,
                             unsigned int num_teams,
                             unsigned int thread_limit, 
                             unsigned int flags),
                   NO_RETURN,
                   ENTRY_PROBE_ARGS(),
                   REAL_SYMBOL_ARGS(fn, data, num_teams, thread_limit, flags),
                   EXIT_PROBE_ARGS(),
                   DEBUG_ARGS("fn=%p data=%p num_teams=%u thread_limit=%u flags=%u", 
                             fn, data, num_teams, thread_limit, flags));


/*******************/ 
/* libgomp/target.c */
/*******************/

/**
 * GOMP_target
 * Avail: GCC >= 4.9
 * 
 * Triggers with a #pragma omp target
 * 
 * Code transformation:
 * <<<
 * >>>
 */
XTR_WRAP_GOMP(GOMP_target,
              PROTOTYPE(int device,
                        void (*fn)(void *),
                        void *unused,
                        size_t mapnum,
                        void **hostaddrs,
                        size_t *sizes, 
                        unsigned char *kinds),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(device, fn, unused, mapnum, hostaddrs, sizes, kinds),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("device=%d fn=%p unused=%p mapnum=%zu hostaddrs=%p sizes=%zu kinds=%p", 
                          device, fn, unused, mapnum, hostaddrs, sizes, kinds));

XTR_WRAP_GOMP(GOMP_target_data,
              PROTOTYPE(int device,
                        void *unused,
                        size_t mapnum,
                        void **hostaddrs,
                        size_t *sizes, 
                        unsigned char *kinds),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(device, unused, mapnum, hostaddrs, sizes, kinds),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("device=%d unused=%p mapnum=%zu hostaddrs=%p sizes=%zu kinds=%p", 
                          device, mapnum, hostaddrs, sizes, kinds));

XTR_WRAP_GOMP(GOMP_target_end_data,
              PROTOTYPE(),
	      NO_RETURN,
	      ENTRY_PROBE_ARGS(),
	      REAL_SYMBOL_ARGS(),
	      EXIT_PROBE_ARGS(),
	      DEBUG_ARGS(""));

XTR_WRAP_GOMP(GOMP_target_update,
              PROTOTYPE(int device,
                        void *unused,
			size_t mapnum,
			void **hostaddrs,
			size_t *sizes,
			unsigned char *kinds),
              NO_RETURN,
	      ENTRY_PROBE_ARGS(),
	      REAL_SYMBOL_ARGS(device, unused, mapnum, hostaddrs, sizes, kinds),
	      EXIT_PROBE_ARGS(),
	      DEBUG_ARGS("device=%d unused=%p mapnum=%zu hostaddrs=%p sizes=%zu kinds=%p",
                          device, unused, mapnum, hostaddrs, sizes, kinds));

XTR_WRAP_GOMP(GOMP_target_enter_exit_data,
              PROTOTYPE(int device,
                        size_t mapnum,
			void **hostaddrs,
			size_t *sizes,
			unsigned short *kinds,
			unsigned int flags,
			void **depend),
              NO_RETURN,
	      ENTRY_PROBE_ARGS(),
	      REAL_SYMBOL_ARGS(device, mapnum, hostaddrs, sizes, kinds, flags, depend),
	      EXIT_PROBE_ARGS(),
	      DEBUG_ARGS("device=%d mapnum=%zu hostaddrs=%p sizes=%zu kinds=%hu flags=%u depend=%p",
                          device, mapnum, hostaddrs, sizes, kinds, flags, depend));

XTR_WRAP_GOMP(GOMP_target_ext,
              PROTOTYPE(int device,
                        void (*fn)(void *),
                        size_t mapnum,
                        void **hostaddrs,
                        size_t *sizes, 
                        unsigned short *kinds,
                        unsigned int flags,
                        void **depend,
                        void **args),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(device, fn, mapnum, hostaddrs, sizes, kinds, flags, depend, args),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("device=%d fn=%p mapnum=%zu hostaddrs=%p sizes=%zu kinds=%hu flags=%u depend=%p args=%p", 
                          device, fn, mapnum, hostaddrs, sizes, kinds, flags, depend, args));

XTR_WRAP_GOMP(GOMP_target_data_ext,
              PROTOTYPE(int device,
                        size_t mapnum,
                        void **hostaddrs,
                        size_t *sizes, 
                        unsigned short *kinds),
              NO_RETURN,
              ENTRY_PROBE_ARGS(),
              REAL_SYMBOL_ARGS(device, mapnum, hostaddrs, sizes, kinds),
              EXIT_PROBE_ARGS(),
              DEBUG_ARGS("device=%d mapnum=%zu hostaddrs=%p sizes=%zu kinds=%hu", 
                          device, mapnum, hostaddrs, sizes, kinds));

XTR_WRAP_GOMP(GOMP_target_update_ext,
              PROTOTYPE(int device,
			size_t mapnum,
			void **hostaddrs,
			size_t *sizes,
			unsigned char *kinds,
			unsigned int flags,
			void **depend),
              NO_RETURN,
	      ENTRY_PROBE_ARGS(),
	      REAL_SYMBOL_ARGS(device, mapnum, hostaddrs, sizes, kinds, flags, depend),
	      EXIT_PROBE_ARGS(),
	      DEBUG_ARGS("device=%d mapnum=%zu hostaddrs=%p sizes=%zu kinds=%p flags=%u depend=%p",
                          device, mapnum, hostaddrs, sizes, kinds, flags, depend));

/*****************/
/* OMP API calls */
/*****************/

/** 
 * omp_set_num_threads
 * omp_set_num_threads_
 * 
 * This wrapper does not emmit events, just calls Backend_ChangeNumberOfThreads.
 * This is necessary when we have a call to 'omp_set_num_threads' followed by a 
 * parallel region that does not specify the 'num_threads' clause.
 * See XTR_WRAP_OMP_SET_NUM_THREADS for details. 
 */
XTR_WRAP_OMP_SET_NUM_THREADS(omp_set_num_threads);
XTR_WRAP_OMP_SET_NUM_THREADS_F(omp_set_num_threads_);

/** 
 * omp_set_lock
 * omp_unset_lock
 * omp_set_lock_
 * omp_unset_lock_
 *
 * The following wrappers correspond to implicit omp calls.
 * omp_get_thread_num has been purposedly ommited.
 * When using a fortran compiler we encounter several variatons of these functions
 * due to name mangling. To support all of them we would have to replicate these wrappers.
 * This situation is better solved in the prototype, for this reason we only maintain the ones
 * that were in the previous implementation of the GOMP tracer.
 */
XTR_WRAP_GOMP(omp_set_lock,
              PROTOTYPE(omp_lock_t *lock),
              NO_RETURN,
              ENTRY_PROBE_ARGS(lock),
              REAL_SYMBOL_ARGS(lock),
              EXIT_PROBE_ARGS(lock),
              DEBUG_ARGS("lock=%p", lock));

XTR_WRAP_GOMP(omp_unset_lock,
              PROTOTYPE(omp_lock_t *lock),
              NO_RETURN,
              ENTRY_PROBE_ARGS(lock),
              REAL_SYMBOL_ARGS(lock),
              EXIT_PROBE_ARGS(lock),
              DEBUG_ARGS("lock=%p", lock));

XTR_WRAP_GOMP(omp_set_lock_,
              PROTOTYPE(omp_lock_t *lock),
              NO_RETURN,
              ENTRY_PROBE_ARGS(lock),
              REAL_SYMBOL_ARGS(lock),
              EXIT_PROBE_ARGS(lock),
              DEBUG_ARGS("lock=%p", lock));

XTR_WRAP_GOMP(omp_unset_lock_,
              PROTOTYPE(omp_lock_t *lock),
              NO_RETURN,
              ENTRY_PROBE_ARGS(lock),
              REAL_SYMBOL_ARGS(lock),
              EXIT_PROBE_ARGS(lock),
              DEBUG_ARGS("lock=%p", lock));
