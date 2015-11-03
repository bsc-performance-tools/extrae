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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.5/src/tracer/wrappers/OMP/omp_wrapper.c $
 | @last_commit: $Date: 2014-02-20 16:48:43 +0100 (jue, 20 feb 2014) $
 | @version:     $Revision: 2487 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef OMPT_HELPER_DEFS_H_INCLUDED
#define OMPT_HELPER_DEFS_H_INCLUDED

#if 0

# include OMPT_HEADER_LOCATION

#else

/* These are extracted from the OMPT spec see: http://openmp.org/mp-documents/ompt-tr2.pdf */

#define NEW_OMPT_DEPS /* for new dependences */

typedef uint64_t ompt_thread_id_t;
typedef uint64_t ompt_parallel_id_t;
typedef uint64_t ompt_task_id_t;
typedef uint64_t ompt_wait_id_t;

typedef struct ompt_frame_s {
	void *exit_runtime_frame;     /* next frame is user code */
	void *reenter_runtime_frame;  /* previous frame is user code */
} ompt_frame_t;

typedef enum {

	/* work states (0..15) */
	ompt_state_work_serial = 0x00, /* working outside parallel */
	ompt_state_work_parallel = 0x01, /* working within parallel */
	ompt_state_work_reduction = 0x02, /* performing a reduction */
	
	/* idle (16..31) */
	ompt_state_idle = 0x10, /* waiting for work */
	
	/* overhead states (32..63) */
	ompt_state_overhead = 0x20, /* non-wait overhead */
	
	/* barrier wait states (64..79) */
	ompt_state_wait_barrier = 0x40, /* generic barrier */
	ompt_state_wait_barrier_implicit = 0x41, /* implicit barrier */
	ompt_state_wait_barrier_explicit = 0x42, /* explicit barrier */
	
	/* task wait states (80..95) */
	ompt_state_wait_taskwait = 0x50, /* waiting at a taskwait */
	ompt_state_wait_taskgroup = 0x51, /* waiting at a taskgroup */
	
	/* mutex wait states (96..111) */
	ompt_state_wait_lock = 0x60, /* waiting for lock */
	ompt_state_wait_nest_lock = 0x61, /* waiting for nest lock */
	ompt_state_wait_critical = 0x62, /* waiting for critical */
	ompt_state_wait_atomic = 0x63, /* waiting for atomic */
	ompt_state_wait_ordered = 0x64, /* waiting for ordered */
	
	/* misc (112.127) */
	ompt_state_undefined = 0x70, /* undefined thread state */
	ompt_state_first = 0x71, /* initial enumeration state */

} ompt_state_t;

typedef enum {

	/*--- Mandatory Events ---*/
	ompt_event_parallel_begin = 1, /* parallel create */
	ompt_event_parallel_end = 2, /* parallel exit */
	ompt_event_task_begin = 3, /* task create */
	ompt_event_task_end = 4, /* task destroy */
	ompt_event_thread_begin = 5, /* thread begin */
	ompt_event_thread_end = 6, /* thread end */
	ompt_event_control = 7, /* support control calls */
	ompt_event_runtime_shutdown = 8, /* runtime shutdown */
	
	/*--- Optional Events (blame shifting) ---*/
	ompt_event_idle_begin = 9, /* begin idle state */
	ompt_event_idle_end = 10, /* end idle state */
	ompt_event_wait_barrier_begin = 11, /* begin wait at barrier */
	ompt_event_wait_barrier_end = 12, /* end wait at barrier */
	ompt_event_wait_taskwait_begin = 13, /* begin wait at taskwait */
	ompt_event_wait_taskwait_end = 14, /* end wait at taskwait */
	ompt_event_wait_taskgroup_begin = 15, /* begin wait at taskgroup */
	ompt_event_wait_taskgroup_end = 16, /* end wait at taskgroup */
	ompt_event_release_lock = 17, /* lock release */
	ompt_event_release_nest_lock_last = 18, /* last nest lock release */
	ompt_event_release_critical = 19, /* critical release */
	ompt_event_release_atomic = 20, /* atomic release */
	ompt_event_release_ordered = 21, /* ordered release */
	
	/*--- Optional Events (synchronous events) --- */
	ompt_event_implicit_task_begin = 22, /* implicit task create */
	ompt_event_implicit_task_end = 23, /* implicit task destroy */
	ompt_event_initial_task_begin = 24, /* initial task create */
	ompt_event_initial_task_end = 25, /* initial task destroy */
	ompt_event_task_switch = 26, /* task switch */
	ompt_event_loop_begin = 27, /* task at loop begin */
	ompt_event_loop_end = 28, /* task at loop end */
	ompt_event_sections_begin = 29, /* task at section begin */
	ompt_event_sections_end = 30, /* task at section end */
	ompt_event_single_in_block_begin = 31, /* task at single begin */
	ompt_event_single_in_block_end = 32, /* task at single end */
	ompt_event_single_others_begin = 33, /* task at single begin */
	ompt_event_single_others_end = 34, /* task at single end */
	ompt_event_workshare_begin = 35, /* task at workshare begin */
	ompt_event_workshare_end = 36, /* task at workshare end */
	ompt_event_master_begin = 37, /* task at master begin */
	ompt_event_master_end = 38, /* task at master end */
	ompt_event_barrier_begin = 39, /* task at barrier begin */
	ompt_event_barrier_end = 40, /* task at barrier end */
	ompt_event_taskwait_begin = 41, /* task at taskwait begin */
	ompt_event_taskwait_end = 42, /* task at task wait end */
	ompt_event_taskgroup_begin = 43, /* task at taskgroup begin */
	ompt_event_taskgroup_end = 44, /* task at taskgroup end */
	ompt_event_release_nest_lock_prev = 45, /* prev nest lock release */
	ompt_event_wait_lock = 46, /* lock wait */
	ompt_event_wait_nest_lock = 47, /* nest lock wait */
	ompt_event_wait_critical = 48, /* critical wait */
	ompt_event_wait_atomic = 49, /* atomic wait */
	ompt_event_wait_ordered = 50, /* ordered wait */
	ompt_event_acquired_lock = 51, /* lock acquired */
	ompt_event_acquired_nest_lock_first = 52, /* 1st nest lock acquired */
	ompt_event_acquired_nest_lock_next = 53, /* next nest lock acquired */
	ompt_event_acquired_critical = 54, /* critical acquired */
	ompt_event_acquired_atomic = 55, /* atomic acquired */
	ompt_event_acquired_ordered = 56, /* ordered acquired */
	ompt_event_init_lock = 57, /* lock init */
	ompt_event_init_nest_lock = 58, /* nest lock init */
	ompt_event_destroy_lock = 59, /* lock destruction */
	ompt_event_destroy_nest_lock = 60, /* nest lock destruction */
	ompt_event_flush = 61, /* after executing flush */
#if !defined(NEW_OMPT_DEPS)
	ompt_event_dependence = 62 /* when a dependece is found, MB project */
#else
	ompt_event_task_dependences = 70, /* new task-dependences */
	ompt_event_task_blocking_dependence = 71 /* blocking task due to dependence */
#endif
} ompt_event_t;

typedef void (*ompt_interface_fn_t) (void);

typedef void (*ompt_callback_t) (void);

typedef enum ompt_thread_type_e {
	ompt_thread_initial = 1,
	ompt_thread_worker  = 2,
	ompt_thread_other   = 3
} ompt_thread_type_t;

typedef ompt_interface_fn_t (*ompt_function_lookup_t) (const char *entry_point);

int ompt_initialize (ompt_function_lookup_t lookup,
	const char *runtime_version,
	unsigned int ompt_ompt_version);

/* callback management */
int ompt_set_callback( /* register a callback for an event */
	ompt_event_t event, /* the event of interest */
	ompt_callback_t callback /* function pointer for the callback */
);

int ompt_get_callback( /* return the current callback for an event (if any) */
	ompt_event_t event, /* the event of interest */
	ompt_callback_t *callback /* pointer to receive the return value */
);

/* state inquiry */
int ompt_enumerate_state( /* extract the set of states supported */
	ompt_state_t current_state, /* current state in the enumeration */
	ompt_state_t *next_state, /* next state in the enumeration */
	const char **next_state_name /* string description of next state */
);

/* thread inquiry */
ompt_thread_id_t ompt_get_thread_id( /* identify the current thread */
	void
);

ompt_state_t ompt_get_state( /* get the state for a thread */
	ompt_wait_id_t *wait_id /* for wait states: identify what awaited */
);

void * ompt_get_idle_frame( /* identify the idle frame (if any) for a thread */
	void
);

/* parallel region inquiry */
ompt_parallel_id_t ompt_get_parallel_id( /* identify a parallel region */
	int ancestor_level /* how many levels the ancestor is removed from the current region */
);

int ompt_get_parallel_team_size( /* query # threads in a parallel region */
	int ancestor_level /* how many levels the ancestor is removed from the current region */
);

/* task inquiry */
ompt_task_id_t *ompt_get_task_id( /* identify a task */
	int depth /* how many levels removed from the current task */
);

ompt_frame_t *ompt_get_task_frame(
	int depth /* how many levels removed from the current task */
);

#if !defined(NEW_OMPT_DEPS)

typedef enum {
   ompt_dependence_raw = 1,
   ompt_dependence_war = 2,
   ompt_dependence_waw = 3 
} ompt_dependence_type_t;

#else

typedef enum ompt_task_dependence_flag_e {
	ompt_task_dependence_type_out   = 1,
	ompt_task_dependence_type_in    = 2,
	ompt_task_dependence_type_inout = 3
} ompt_task_dependence_flag_t;

typedef struct {
	void * variable_addr;
	size_t len;
	uint32_t flags;
} ompt_task_dependence_t;

#endif

#endif 

#endif /* OMPT_HELPER_DEFS_H_INCLUDED */

