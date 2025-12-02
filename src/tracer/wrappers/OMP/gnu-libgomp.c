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

#include "common.h"

#ifdef HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef  __USE_GNU
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if !defined(HAVE__SYNC_FETCH_AND_ADD)
# ifdef HAVE_PTHREAD_H
#  include <pthread.h>
# endif
#endif
#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif

#include "gnu-libgomp.h"
#include "omp-events.h"
#include "omp-common.h"
#include "omp-probe.h"
#include "wrapper.h"
#include <omp.h>

#include "xalloc.h"

//#define DEBUG
#define GOMP_API_3_1 "3.1"
#define GOMP_API_4_0 "4.0"
#define GOMP_API_4_5 "4.5"

char *__GOMP_version = NULL;

/*
 * In case the constructor initialization didn't trigger
 * or the symbols couldn't be found, retry hooking.
 */
#define RECHECK_INIT(real_fn_ptr)                                      \
{                                                                      \
  if (real_fn_ptr == NULL)                                             \
  {                                                                    \
    fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL                 \
		                 "%s: WARNING! %s is a NULL pointer. "             \
		                 "Did the initialization of this module trigger? " \
		                 "Retrying initialization...\n",                   \
		                 THREAD_LEVEL_VAR, __func__, #real_fn_ptr);        \
		_extrae_gnu_libgomp_init (TASKID);                                 \
  }                                                                    \
}

#if defined(PIC)

/**************************************************************/
/***** Added (or changed) in OpenMP 3.1 or prior versions *****/
/**************************************************************/

static int gnu_libgomp_get_hook_points (int rank);

static void (*GOMP_atomic_start_real)(void) = NULL;
static void (*GOMP_atomic_end_real)(void) = NULL;

static void (*GOMP_barrier_real)(void) = NULL;

static void (*GOMP_critical_start_real)(void) = NULL;
static void (*GOMP_critical_end_real)(void) = NULL;
static void (*GOMP_critical_name_start_real)(void**) = NULL;
static void (*GOMP_critical_name_end_real)(void**) = NULL;

static int (*GOMP_loop_static_start_real)(long,long,long,long,long*,long*) = NULL;
static int (*GOMP_loop_dynamic_start_real)(long,long,long,long,long*,long*) = NULL;
static int (*GOMP_loop_guided_start_real)(long,long,long,long,long*,long*) = NULL;
static int (*GOMP_loop_runtime_start_real)(long,long,long,long,long*,long*) = NULL;

static int (*GOMP_loop_static_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_dynamic_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_guided_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_runtime_next_real)(long*,long*) = NULL;

static int (*GOMP_loop_ordered_static_start_real)(long, long, long, long, long *, long *) = NULL;
static int (*GOMP_loop_ordered_dynamic_start_real)(long, long, long, long, long *, long *) = NULL;
static int (*GOMP_loop_ordered_guided_start_real)(long, long, long, long, long *, long *) = NULL;
static int (*GOMP_loop_ordered_runtime_start_real)(long, long, long, long, long *, long *) = NULL;

static int (*GOMP_loop_ordered_static_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_ordered_dynamic_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_ordered_guided_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_ordered_runtime_next_real)(long*,long*) = NULL;

static void (*GOMP_parallel_loop_static_start_real)(void*, void*,unsigned, long, long, long, long) = NULL;
static void (*GOMP_parallel_loop_dynamic_start_real)(void*, void*,unsigned, long, long, long, long) = NULL;
static void (*GOMP_parallel_loop_guided_start_real)(void*, void*,unsigned, long, long, long, long) = NULL;
static void (*GOMP_parallel_loop_runtime_start_real)(void*, void*,unsigned, long, long, long) = NULL;

static void (*GOMP_loop_end_real)(void) = NULL;
static void (*GOMP_loop_end_nowait_real)(void) = NULL;

static void (*GOMP_ordered_start_real)(void) = NULL;
static void (*GOMP_ordered_end_real)(void) = NULL;

static void (*GOMP_parallel_start_real)(void*,void*,unsigned) = NULL;
static void (*GOMP_parallel_end_real)(void) = NULL;

static void (*GOMP_parallel_sections_start_real)(void*,void*,unsigned,unsigned) = NULL;
static unsigned (*GOMP_sections_start_real)(unsigned) = NULL;
static unsigned (*GOMP_sections_next_real)(void) = NULL;
static void (*GOMP_sections_end_real)(void) = NULL;
static void (*GOMP_sections_end_nowait_real)(void) = NULL;

static unsigned (*GOMP_single_start_real)(void) = NULL;

static void (*GOMP_taskwait_real)(void) = NULL;
static void (*GOMP_taskyield_real)(void) = NULL;

/********************************************/
/***** Added (or changed) in OpenMP 4.0 *****/
/********************************************/

static void (*GOMP_parallel_real)(void*,void*,unsigned,unsigned int) = NULL;

static void (*GOMP_parallel_loop_static_real)(void*, void*, unsigned, long, long, long, long, unsigned) = NULL;
static void (*GOMP_parallel_loop_dynamic_real)(void*, void*, unsigned, long, long, long, long, unsigned) = NULL;
static void (*GOMP_parallel_loop_guided_real)(void*, void*, unsigned, long, long, long, long, unsigned) = NULL;
static void (*GOMP_parallel_loop_runtime_real)(void*, void*, unsigned, long, long, long, unsigned) = NULL;

static void (*GOMP_parallel_sections_real)(void *, void *, unsigned, unsigned, unsigned) = NULL;

static void (*GOMP_taskgroup_start_real)(void) = NULL;
static void (*GOMP_taskgroup_end_real)(void) = NULL;

static void (*GOMP_target_real)(int, void *, const void *, size_t, void **, size_t *, unsigned char *) = NULL;
static void (*GOMP_target_data_real)(int, const void *, size_t, void **, size_t *, unsigned char *) = NULL;
static void (*GOMP_target_end_data_real)(void) = NULL;
static void (*GOMP_target_update_real)(int, const void *, size_t, void **, size_t *, unsigned char *) = NULL;

/********************************************/
/***** Added (or changed) in OpenMP 4.5 *****/
/********************************************/

// Appeared in OpenMP 3.1 but increased the #parameters in 4.0 and later in 4.5
static void (*GOMP_task_real)(void*,void*,void*,long,long,int,unsigned,...) = NULL;
static void (*GOMP_taskloop_real)(void*,void*,void*,long,long,unsigned,unsigned long,int,long,long,long) = NULL;
static void (*GOMP_taskloop_ull_real)(void*,void*,void*,long,long,unsigned,unsigned long,int,unsigned long long,unsigned long long,unsigned long long) = NULL;

static int (*GOMP_loop_doacross_static_start_real)(unsigned, long *, long, long *, long *) = NULL;
static int (*GOMP_loop_doacross_dynamic_start_real)(unsigned, long *, long, long *, long *) = NULL;
static int (*GOMP_loop_doacross_guided_start_real)(unsigned, long *, long, long *, long *) = NULL;
static int (*GOMP_loop_doacross_runtime_start_real)(unsigned, long *, long *, long *) = NULL;
static void (*GOMP_doacross_post_real)(long *) = NULL;
static void (*GOMP_doacross_wait_real)(long, ...) = NULL;

static void (*GOMP_target_ext_real)(int, void *, size_t, void **, size_t *, unsigned short *, unsigned int, void **, void **) = NULL;
static void (*GOMP_target_data_ext_real)(int, size_t, void **, size_t *, unsigned short *) = NULL;
static void (*GOMP_target_end_data_ext_real)(void) = NULL;
static void (*GOMP_target_update_ext_real)(int, size_t, void **, size_t *, unsigned short *, unsigned int, void **) = NULL;
static void (*GOMP_target_enter_exit_data_real)(int, size_t, void **, size_t *, unsigned short *, unsigned int, void **) = NULL;



/******************************************************************************\
 *                                                                            *
 *                                  HELPERS                                   *
 *                                                                            *
 ******************************************************************************
 * The following structures are used to encapsulate the parameters of the     *
 * runtime routines, allowing us to modify them so as to change the outlined  *
 * routines to our own methods to emit instrumentation. Then we retrieve the  *
 * real data from the helpers to callback the real outlined functions.        *
\******************************************************************************/

/*
 * __GOMP_helpers is a structure that contains an array of all active data helpers.
 * The array of the queue can be increased from DEFAULT_OPENMP_HELPERS to the 
 * value set by the EXTRAE_OPENMP_HELPERS environment variable.
 */
pthread_mutex_t __GOMP_helpers_mtx = PTHREAD_MUTEX_INITIALIZER;
struct helpers_queue_t *__GOMP_helpers = NULL;

/**
 * preallocate_GOMP_helpers
 *
 * Allocates the structure __GOMP_helpers to hold a queue of active data helpers.
 */
static void preallocate_GOMP_helpers()
{
	int num_helpers = 0;
	char *env_helpers = NULL;

	pthread_mutex_lock(&__GOMP_helpers_mtx);                                      

	if (__GOMP_helpers == NULL)                                                      
	{                                                                             
		__GOMP_helpers = (struct helpers_queue_t *)xmalloc(sizeof(struct helpers_queue_t));                                                                        

		/*
		 * If the environment variable ENV_VAR_EXTRAE_OPENMP_HELPERS is defined, this
		 * will be the size of the queue. Otherwise, DEFAULT_OPENMP_HELPERS is used.
		 */
		env_helpers = getenv(ENV_VAR_EXTRAE_OPENMP_HELPERS);                        
		if (env_helpers != NULL)                                                    
		{                                                                           
			num_helpers = atoi(env_helpers);                                          
		}                                                                           
		if (num_helpers <= 0)                                                       
		{                                                                           
			num_helpers = DEFAULT_OPENMP_HELPERS;                                     
		}                                                                           

#if defined(DEBUG)                                                              
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "Allocating %d data helpers\n", THREAD_LEVEL_VAR, num_helpers);
#endif                                                                          

		__GOMP_helpers->current_helper = 0;                                           
		__GOMP_helpers->max_helpers = num_helpers;                                    
		__GOMP_helpers->queue = (struct parallel_helper_t *)xmalloc(sizeof(struct parallel_helper_t) * num_helpers);                                                                          
	}                                                                             

	pthread_mutex_unlock(&__GOMP_helpers_mtx);
}


/**
 * __GOMP_new_helper
 *
 * Registers a new data helper in the __GOMP_helpers queue. When we run out of 
 * helpers, we start reusing. As long as the corresponding parallel region already
 * finished, there will be no problems. But if that parallel region is still
 * active, we'll be corrupting the pointers to fn and data. A warning is shown
 * the first time the helpers are reused, and if the application is corrupted, 
 * we need to increase the number of helpers setting EXTRAE_OPENMP_HELPERS.
 *
 * @param fn The pointer to the real outlined function or task.
 * @param data The pointer to the data passed to the real routine.
 *
 * @return The pointer to the queue slot where the data helper is stored.
 */
void *__GOMP_new_helper(void (*fn)(void *), void *data)
{
	int idx = 0;
	void *helper_ptr = NULL;
	static int warning_displayed = 0;

	pthread_mutex_lock(&__GOMP_helpers_mtx);

	/* Pick a slot in the queue */
	idx = __GOMP_helpers->current_helper;
	__GOMP_helpers->current_helper = (__GOMP_helpers->current_helper + 1) % __GOMP_helpers->max_helpers;

	pthread_mutex_unlock(&__GOMP_helpers_mtx);

	/* Save the pointers to fn and data */
	__GOMP_helpers->queue[idx].fn = fn;
	__GOMP_helpers->queue[idx].data = data;

	/* Return the pointer to the slot for this data helper */
	helper_ptr = &(__GOMP_helpers->queue[idx]);

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__GOMP_new_helper: Registering helper #%d helper_ptr=%p fn=%p data=%p\n", THREAD_LEVEL_VAR, idx, helper_ptr, fn, data);
#endif

	if (__GOMP_helpers->current_helper < idx)
	{
		/*
		 * Display a warning (once) when we start reusing queue slots for helpers. 
		 * Could appear more than once in the event that two concurrent threads 
		 * evaluate warning_displayed simultaneously, but this is extremely
		 * unlikely and it's better not to use a mutex here to minimize overhead.
		 */
		if (!warning_displayed)
		{
			fprintf (stderr, PACKAGE_NAME": WARNING! I have run out of allocations for data helpers. If the application starts crashing or producing wrong results, please try increasing %s over %d until this warning disappears\n", ENV_VAR_EXTRAE_OPENMP_HELPERS, __GOMP_helpers->max_helpers);
			warning_displayed = 1;
		}
	}

	return helper_ptr;
}

/*
 * When we enter a GOMP_parallel we have to store the par_uf later used in the
 * loop and loop_ordered calls after opening parallelism (current level + 1).
 * FIXME: This doesn't support nested because multiple threads can open a 2nd
 * level of parallelism, and they would all overwrite the same position of the
 * array as they're all in the same nesting level. Can we index by thread id?
 * Previously, this was just a global variable.
 */
static void * __GOMP_parallel_uf[MAX_NESTING_LEVEL];

void SAVE_PARALLEL_UF(void *par_uf)
{
  int level = omp_get_level();
  CHECK_NESTING_LEVEL(level);
  __GOMP_parallel_uf[level] = par_uf;
}

void * RETRIEVE_PARALLEL_UF()
{
  int level = omp_get_level();
  CHECK_NESTING_LEVEL(level-1);
  return __GOMP_parallel_uf[level-1];
}

/*
 * Counter to enumerate tasks executed from GOMP_task 
 */
static volatile long long __GOMP_task_ctr = 1;
#if !defined(HAVE__SYNC_FETCH_AND_ADD)
static pthread_mutex_t __GOMP_task_ctr_mtx = PTHREAD_MUTEX_INITIALIZER;
#endif

/*
 * Counter to enumerate tasks executed from GOMP_taskloop
 */
static volatile long long __GOMP_taskloop_ctr = 1;
#if !defined(HAVE__SYNC_FETCH_AND_ADD)
static pthread_mutex_t __GOMP_taskloop_ctr_mtx = PTHREAD_MUTEX_INITIALIZER;
#endif

/*
 * Instrumentation of doacross is split in several routines. In the *_start 
 * routines (e.g. GOMP_loop_doacross_static_start) we need to save the parameter
 * 'ncounts', which is later needed in the routine GOMP_doacross_wait. Since the
 * latter does not receive any parameter that we can replace with a data helper,
 * what we do is to store ncounts in the thread TLS. This can be done because 
 * all threads execute the *_start routine. This parameter is saved per nesting
 * level, to support nowait clauses and nested parallelism. 
 */
static __thread unsigned __GOMP_doacross_ncounts[MAX_NESTING_LEVEL]; 

/** 
 * SAVE_DOACROSS_NCOUNTS
 *
 * Save the ncounts parameter from the doacross *_start routine in the thread
 * TLS to be retrieved later in GOMP_doacross_wait. This is stored in an array
 * indexed per current nesting level, in order to allow nowait clauses and
 * nested loops.
 *
 * @param ncounts The parameter to save in the TLS
 */
void SAVE_DOACROSS_NCOUNTS(unsigned ncounts)
{
	int level = omp_get_level();
	CHECK_NESTING_LEVEL(level);
  __GOMP_doacross_ncounts[level] = ncounts;
}

/** 
 * RETRIEVE_DOACROSS_NCOUNTS
 * 
 * Retrieve the ncounts parameter from the thread TLS (previously stored in the
 * GOMP_loop_doacross_*_start function). This is stored in an array indexed per
 * current nesting level, in order to allow nowait clauses and nested loops. 
 *
 * @return the ncounts parameter previously saved in the TLS.
 */
unsigned RETRIEVE_DOACROSS_NCOUNTS()
{
	int level = omp_get_level();
	CHECK_NESTING_LEVEL(level);
  return __GOMP_doacross_ncounts[level];
}


/******************************************************************************\
 *                                                                            *
 *                                 CALLBACKS                                  *
 *                                                                            *
 ******************************************************************************
 * The following callme_* routines replace the real outlined functions and    *
 * tasks, so that when the runtime would actually execute the task, it runs   *
 * our function instead where we emit events that mark when the task is       *
 * executed. From these functions we retrieve the pointer to the original     *
 * routine from the various data helpers, and invoke it.                      *
\******************************************************************************/ 

static void callme_parsections (void *parsections_helper_ptr)
{
	Backend_Enter_Instrumentation ();
	struct parallel_helper_t *parsections_helper = parsections_helper_ptr;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "callme_parsections enter: parsections_helper=%p fn=%p data=%p\n", THREAD_LEVEL_VAR, parsections_helper, parsections_helper->fn, parsections_helper->data);
#endif

	if ((parsections_helper == NULL) || (parsections_helper->fn == NULL))
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! Invalid initialization of 'parsections_helper'\n");
		exit (-1);
	}

	Probe_OpenMP_UF_Entry (parsections_helper->fn);
	Backend_Leave_Instrumentation (); /* We're about to execute user code */
	parsections_helper->fn (parsections_helper->data);
	Backend_Enter_Instrumentation (); /* We're about to execute OpenMP code */
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void callme_pardo (void *pardo_helper_ptr)
{
	Backend_Enter_Instrumentation ();
	struct parallel_helper_t *pardo_helper = pardo_helper_ptr;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "callme_pardo enter: pardo_helper=%p fn=%p data=%p\n", THREAD_LEVEL_VAR, pardo_helper, pardo_helper->fn, pardo_helper->data);
#endif

	if ((pardo_helper == NULL) || (pardo_helper->fn == NULL))
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! Invalid initialization of 'pardo_helper'\n");
		exit (-1);
	}

	Probe_OpenMP_DO_Entry ();
	Probe_OpenMP_UF_Entry (pardo_helper->fn);
	Backend_Leave_Instrumentation (); /* We're about to execute user code */
	pardo_helper->fn (pardo_helper->data);
	Backend_Enter_Instrumentation (); /* We're about to execute OpenMP code */
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void callme_par (void *par_helper_ptr)
{
	Backend_Enter_Instrumentation ();
	struct parallel_helper_t *par_helper = par_helper_ptr;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "callme_par enter: par_helper=%p fn=%p data=%p\n", THREAD_LEVEL_VAR, par_helper, par_helper->fn, par_helper->data);
#endif

	if ((par_helper == NULL) || (par_helper->fn == NULL))
	{
		fprintf (stderr, PACKAGE_NAME": ERROR! Invalid initialization of 'par_helper'\n");
		exit (-1);
	}

	Probe_OpenMP_UF_Entry (par_helper->fn);
	Backend_Leave_Instrumentation (); /* We're about to execute user code */
	par_helper->fn (par_helper->data);
	Backend_Enter_Instrumentation (); /* We're about to execute OpenMP code */
	Probe_OpenMP_UF_Exit ();

#if defined(DEBUG)
	  fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "callme_par ends\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

static void callme_task (void *task_helper_ptr)
{
	Backend_Enter_Instrumentation ();
	struct task_helper_t *task_helper = *(struct task_helper_t **)task_helper_ptr;

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "callme_task enter: task_helper_ptr=%p\n", THREAD_LEVEL_VAR, task_helper_ptr);
#endif

	if (task_helper != NULL)
	{
		Probe_OpenMP_TaskUF_Entry (task_helper->fn);
		Extrae_OpenMP_TaskID (task_helper->counter, XTR_TASK_EXECUTION);

		Backend_Leave_Instrumentation ();
		task_helper->fn (task_helper->data);
		Backend_Enter_Instrumentation ();

		if (task_helper->buf != NULL)
			xfree(task_helper->buf);
		xfree(task_helper);

		Extrae_OpenMP_Notify_NewExecutedTask();
		Probe_OpenMP_TaskUF_Exit ();
	}
	Backend_Leave_Instrumentation ();
}


static void callme_taskloop(void *data_trailer)
{
	Backend_Enter_Instrumentation ();
	struct taskloop_helper_t *taskloop_helper = NULL;

	// Search for the magic number 0xdeadbeef to locate the helper
	int i = sizeof(void *), arg_size = 0;
	while (*(void **)(data_trailer + i) != (void *)0xdeadbeef)
	{
		i ++;
	}
	arg_size = i;
	taskloop_helper = data_trailer + arg_size;
	
	Backend_Leave_Instrumentation ();
	void (*fn)(void *) = taskloop_helper->fn;
	Backend_Enter_Instrumentation ();

	Probe_OpenMP_TaskUF_Entry (fn);
	Extrae_OpenMP_TaskLoopID (taskloop_helper->id);

	fn (data_trailer);

        Extrae_OpenMP_Notify_NewExecutedTask();
	Probe_OpenMP_TaskUF_Exit ();

	Backend_Leave_Instrumentation ();
}

/******************************************************************************\
 *                                                                            *
 *                                WRAPPERS                                    *
 *                                                                            *
\******************************************************************************/

/**************************************************************/
/***** Added (or changed) in OpenMP 3.1 or prior versions *****/
/**************************************************************/

void GOMP_atomic_start (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_atomic_start enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_atomic_start_real);
#endif

	RECHECK_INIT(GOMP_atomic_start_real);

	if (TRACE(GOMP_atomic_start_real))
	{
		Probe_OpenMP_Unnamed_Lock_Entry();
		GOMP_atomic_start_real();
		Probe_OpenMP_Unnamed_Lock_Exit();
	}
	else if (GOMP_atomic_start_real != NULL)
	{
		GOMP_atomic_start_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_atomic_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_atomic_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_atomic_end (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_atomic_end enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_atomic_end_real);
#endif

	RECHECK_INIT(GOMP_atomic_end_real);

	if (TRACE(GOMP_atomic_end_real))
	{
		Probe_OpenMP_Unnamed_Unlock_Entry();
		GOMP_atomic_end_real ();
		Probe_OpenMP_Unnamed_Unlock_Exit();
	}
	else if (GOMP_atomic_end_real != NULL)
	{
		GOMP_atomic_end_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_atomic_end: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_atomic_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_barrier (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_barrier enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_barrier_real);
#endif

	RECHECK_INIT(GOMP_barrier_real);

	if (TRACE(GOMP_barrier_real))
	{
		Probe_OpenMP_Barrier_Entry ();
		GOMP_barrier_real ();
		Probe_OpenMP_Barrier_Exit ();
	}
	else if (GOMP_barrier_real != NULL)
	{
		GOMP_barrier_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_barrier: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_barrier exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_critical_start (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_start enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_critical_start_real);
#endif

	RECHECK_INIT(GOMP_critical_start_real);

	if (TRACE(GOMP_critical_start_real))
	{
		Probe_OpenMP_Unnamed_Lock_Entry();
		GOMP_critical_start_real();
		Probe_OpenMP_Unnamed_Lock_Exit();
	}
	else if (GOMP_critical_start_real != NULL)
	{
		GOMP_critical_start_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_critical_end (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_end enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_critical_end_real);
#endif

	RECHECK_INIT(GOMP_critical_end_real);

	if (TRACE(GOMP_critical_end_real))
	{
		Probe_OpenMP_Unnamed_Unlock_Entry();
		GOMP_critical_end_real ();
		Probe_OpenMP_Unnamed_Unlock_Exit();
	}
	else if (GOMP_critical_end_real != NULL)
	{
		GOMP_critical_end_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_end: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_critical_name_start (void **pptr)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_name_start enter: @=%p args=(%p)\n", THREAD_LEVEL_VAR, GOMP_critical_name_start_real, pptr);
#endif

	RECHECK_INIT(GOMP_critical_name_start_real);

	if (TRACE(GOMP_critical_name_start_real))
	{
		Probe_OpenMP_Named_Lock_Entry();
		GOMP_critical_name_start_real (pptr);
		Probe_OpenMP_Named_Lock_Exit(pptr);
	}
	else if (GOMP_critical_name_start_real != NULL)
	{
		GOMP_critical_name_start_real (pptr);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_name_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_name_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_critical_name_end (void **pptr)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_name_end enter: @=%p args=(%p)\n", THREAD_LEVEL_VAR, GOMP_critical_name_end_real, pptr);
#endif

	RECHECK_INIT(GOMP_critical_name_end_real);

	if (TRACE(GOMP_critical_name_end_real))
	{
		Probe_OpenMP_Named_Unlock_Entry(pptr);
		GOMP_critical_name_end_real (pptr);
		Probe_OpenMP_Named_Unlock_Exit();
	}
	else if (GOMP_critical_name_end_real != NULL)
	{
		GOMP_critical_name_end_real (pptr);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_name_end: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_critical_name_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

int GOMP_loop_static_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;

	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_static_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_static_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_static_start_real);

	if (TRACE(GOMP_loop_static_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_static_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_static_start_real != NULL)
	{
		res = GOMP_loop_static_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_static_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_static_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_dynamic_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_dynamic_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_dynamic_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_dynamic_start_real);

	if (TRACE(GOMP_loop_dynamic_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_dynamic_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_dynamic_start_real != NULL)
	{
		res = GOMP_loop_dynamic_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_dynamic_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_dynamic_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_guided_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_guided_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_guided_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_guided_start_real);

	if (TRACE(GOMP_loop_guided_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_guided_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_guided_start_real != NULL)
	{
		res = GOMP_loop_guided_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_guided_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_guided_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_runtime_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_runtime_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_runtime_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_runtime_start_real);

	if (TRACE(GOMP_loop_runtime_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_runtime_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_runtime_start_real != NULL)
	{
		res = GOMP_loop_runtime_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_runtime_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_runtime_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_static_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_static_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_static_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_static_next_real);

	if (TRACE(GOMP_loop_static_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_static_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_static_next_real != NULL)
	{
		res = GOMP_loop_static_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_static_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_static_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_dynamic_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_dynamic_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_dynamic_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_dynamic_next_real);

	if (TRACE(GOMP_loop_dynamic_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_dynamic_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_dynamic_next_real != NULL)
	{
		res = GOMP_loop_dynamic_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_dynamic_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_dynamic_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_guided_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_guided_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_guided_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_guided_next_real);

	if (TRACE(GOMP_loop_guided_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_guided_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_guided_next_real != NULL)
	{
		res = GOMP_loop_guided_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_guided_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_guided_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_runtime_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_runtime_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_runtime_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_runtime_next_real);

	if (TRACE(GOMP_loop_runtime_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_runtime_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_runtime_next_real != NULL)
	{
		res = GOMP_loop_runtime_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_runtime_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_runtime_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_static_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_static_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_static_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_static_start_real);

	if (TRACE(GOMP_loop_ordered_static_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_static_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_ordered_static_start_real != NULL)
	{
		res = GOMP_loop_ordered_static_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_static_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_static_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_dynamic_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_dynamic_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_dynamic_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_dynamic_start_real);

	if (TRACE(GOMP_loop_ordered_dynamic_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_dynamic_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_ordered_dynamic_start_real != NULL)
	{
		res = GOMP_loop_ordered_dynamic_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_dynamic_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_dynamic_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_guided_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_guided_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_guided_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_guided_start_real);

	if (TRACE(GOMP_loop_ordered_guided_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_guided_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_ordered_guided_start_real != NULL)
	{
		res = GOMP_loop_ordered_guided_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_guided_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_guided_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_runtime_start (long start, long end, long incr, long chunk_size, long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_runtime_start enter: @=%p args=(%ld %ld %ld %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_runtime_start_real, start, end, incr, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_runtime_start_real);

	if (TRACE(GOMP_loop_ordered_runtime_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_runtime_start_real (start, end, incr, chunk_size, istart, iend);
		Probe_OpenMP_UF_Entry (RETRIEVE_PARALLEL_UF());
	}
	else if (GOMP_loop_ordered_runtime_start_real != NULL)
	{
		res = GOMP_loop_ordered_runtime_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_runtime_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_runtime_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_static_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_static_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_static_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_static_next_real);

	if (TRACE(GOMP_loop_ordered_static_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_ordered_static_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_ordered_static_next_real != NULL)
	{
		res = GOMP_loop_ordered_static_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_static_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_static_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_dynamic_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_dynamic_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_dynamic_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_dynamic_next_real);

	if (TRACE(GOMP_loop_ordered_dynamic_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_ordered_dynamic_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_ordered_dynamic_next_real != NULL)
	{
		res = GOMP_loop_ordered_dynamic_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_dynamic_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_dynamic_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_guided_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_guided_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_guided_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_guided_next_real);

	if (TRACE(GOMP_loop_ordered_guided_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_ordered_guided_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_ordered_guided_next_real != NULL)
	{
		res = GOMP_loop_ordered_guided_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_guided_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_guided_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_ordered_runtime_next (long *istart, long *iend)
{
	int res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_runtime_next enter: @=%p args=(%p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_ordered_runtime_next_real, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_ordered_runtime_next_real);

	if (TRACE(GOMP_loop_ordered_runtime_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_ordered_runtime_next_real (istart, iend);
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_ordered_runtime_next_real != NULL)
	{
		res = GOMP_loop_ordered_runtime_next_real (istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_runtime_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_ordered_runtime_next exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

void
GOMP_parallel_loop_static_start(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr, long chunk_size)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_static_start enter: @=%p args=(%p %p %u %ld %ld %ld %ld)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_static_start_real, fn, data, num_threads, start, end, incr, chunk_size);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_static_start_real);

	if (TRACE(GOMP_parallel_loop_static_start_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_static_start_real(callme_pardo, pardo_helper, num_threads, start, end, incr, chunk_size);
		Backend_Enter_Instrumentation ();

		/*
		 * The master thread continues the execution and then calls fn
		 */
		if (THREADID == 0)
		{
			Probe_OpenMP_DO_Entry ();
			Probe_OpenMP_UF_Entry(fn);
		}

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_static_start_real != NULL)
	{
		GOMP_parallel_loop_static_start_real(fn, data, num_threads, start, end, incr, chunk_size);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_static_start: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_static_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_loop_dynamic_start(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr, long chunk_size)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_dynamic_start enter: @=%p args=(%p %p %u %ld %ld %ld %ld)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_dynamic_start_real, fn, data, num_threads, start, end, incr, chunk_size);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_dynamic_start_real);

	if (TRACE(GOMP_parallel_loop_dynamic_start_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();

		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_dynamic_start_real(callme_pardo, pardo_helper, num_threads, start, end, incr, chunk_size);
		Backend_Enter_Instrumentation ();

		/*
		 * The master thread continues the execution and then calls fn
		 */
		if (THREADID == 0)
		{
			Probe_OpenMP_DO_Entry ();
			Probe_OpenMP_UF_Entry(fn);
		}

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_dynamic_start_real != NULL)
	{
		GOMP_parallel_loop_dynamic_start_real(fn, data, num_threads, start, end, incr, chunk_size);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_dynamic_start: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_dynamic_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_loop_guided_start(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr, long chunk_size)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_guided_start enter: @=%p args=(%p %p %u %ld %ld %ld %ld)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_guided_start_real, fn, data, num_threads, start, end, incr, chunk_size);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_guided_start_real);

	if (TRACE(GOMP_parallel_loop_guided_start_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();

		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_guided_start_real(callme_pardo, pardo_helper, num_threads, start, end, incr, chunk_size);
		Backend_Enter_Instrumentation ();

		/*
		 * The master thread continues the execution and then calls fn
		 */
		if (THREADID == 0)
		{
			Probe_OpenMP_DO_Entry ();
			Probe_OpenMP_UF_Entry(fn);
		}

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_guided_start_real != NULL)
	{
		GOMP_parallel_loop_guided_start_real(fn, data, num_threads, start, end, incr, chunk_size);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_guided_start: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_guided_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_loop_runtime_start(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_runtime_start enter: @=%p args=(%p %p %u %ld %ld %ld)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_runtime_start_real, fn, data, num_threads, start, end, incr);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_runtime_start_real);

	if (TRACE(GOMP_parallel_loop_runtime_start_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_runtime_start_real(callme_pardo, pardo_helper, num_threads, start, end, incr);
		Backend_Enter_Instrumentation ();

		/*
		 * The master thread continues the execution and then calls fn
		 */
		if (THREADID == 0)
		{
			Probe_OpenMP_DO_Entry ();
			Probe_OpenMP_UF_Entry(fn);
		}

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_runtime_start_real != NULL)
	{
		GOMP_parallel_loop_runtime_start_real(fn, data, num_threads, start, end, incr);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_runtime_start: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_runtime_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_loop_end (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_end enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_loop_end_real);
#endif

	RECHECK_INIT(GOMP_loop_end_real);

	if (TRACE(GOMP_loop_end_real))
	{
		Probe_OpenMP_Join_Wait_Entry();
		GOMP_loop_end_real();
		Probe_OpenMP_Join_Wait_Exit();
		Probe_OpenMP_DO_Exit ();	
	}
	else if (GOMP_loop_end_real != NULL)
	{
		GOMP_loop_end_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_end: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_loop_end_nowait (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_end_nowait enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_loop_end_nowait_real);
#endif

	RECHECK_INIT(GOMP_loop_end_nowait_real);

	if (TRACE(GOMP_loop_end_nowait_real))
	{
		Probe_OpenMP_Join_NoWait_Entry();
		GOMP_loop_end_nowait_real();
		Probe_OpenMP_Join_NoWait_Exit();
		Probe_OpenMP_DO_Exit ();	
	}
	else if (GOMP_loop_end_nowait_real != NULL)
	{
		GOMP_loop_end_nowait_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_end_nowait: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_end_nowait exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_ordered_start (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_ordered_start enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_ordered_start_real);
#endif

	RECHECK_INIT(GOMP_ordered_start_real);

	if (TRACE(GOMP_ordered_start_real))
	{
		Probe_OpenMP_Ordered_Wait_Entry();
		GOMP_ordered_start_real ();
		Probe_OpenMP_Ordered_Wait_Exit();
	}
	else if (GOMP_ordered_start_real != NULL)
	{
		GOMP_ordered_start_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_ordered_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_ordered_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_ordered_end (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_ordered_end enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_ordered_end_real);
#endif

	RECHECK_INIT(GOMP_ordered_end_real);

	if (TRACE(GOMP_ordered_end_real))
	{
		Probe_OpenMP_Ordered_Post_Entry();
		GOMP_ordered_end_real();
		Probe_OpenMP_Ordered_Post_Exit();
	}
	else if (GOMP_ordered_end_real != NULL)
	{
		GOMP_ordered_end_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_ordered_end_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_ordered_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_start(void (*fn)(void *), void *data, unsigned num_threads)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_start enter: @=%p args=(%p %p %u)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_start_real, fn, data, num_threads);
#endif

	RECHECK_INIT(GOMP_parallel_start_real);

	if (TRACE(GOMP_parallel_start_real))
	{
		SAVE_PARALLEL_UF(fn);

		void *par_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParRegion_Entry();
		Extrae_OpenMP_EmitTaskStatistics();

		Backend_Leave_Instrumentation ();
		GOMP_parallel_start_real (callme_par, par_helper, num_threads);
		/* GCC/libgomp does not execute callme_par per root thread, emit
		   the required event here - call Backend to get a new time! */
		Backend_Enter_Instrumentation ();

		Probe_OpenMP_UF_Entry(fn);
	}
	else if (GOMP_parallel_start_real != NULL)
	{
		GOMP_parallel_start_real(fn, data, num_threads);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_start: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_end(void)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_end enter: @=%p\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_end_real);
#endif

	RECHECK_INIT(GOMP_parallel_end_real);

	if (TRACE(GOMP_parallel_end_real))
	{
		Probe_OpenMP_UF_Exit();
		GOMP_parallel_end_real();
		Probe_OpenMP_ParRegion_Exit();
		Extrae_OpenMP_EmitTaskStatistics();

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_end_real != NULL)
	{
		GOMP_parallel_end_real();
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_end: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_sections_start(void (*fn)(void *), void *data, unsigned num_threads, unsigned count)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_sections_start enter: @=%p args=(%p %p %u %u)\n",
	    THREAD_LEVEL_VAR, GOMP_sections_start_real, fn, data, num_threads, count);
#endif

	RECHECK_INIT(GOMP_parallel_sections_start_real);

	if (TRACE(GOMP_parallel_sections_start_real))
	{
		void *parsections_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParSections_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_sections_start_real(callme_parsections, parsections_helper, num_threads, count);
		Backend_Enter_Instrumentation ();

		/* The master thread continues the execution and then calls parsections_helper->fn */
		if (THREADID == 0)
		{
			Probe_OpenMP_UF_Entry(fn);
		}
	}
	else if (GOMP_parallel_sections_start_real != NULL)
	{
		GOMP_parallel_sections_start_real(fn, data, num_threads, count);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_sections_start: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_sections_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

unsigned GOMP_sections_start (unsigned count)
{
	unsigned res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_start enter: @=%p args=(%u)\n", THREAD_LEVEL_VAR, GOMP_sections_start_real, count);
#endif

	RECHECK_INIT(GOMP_sections_start_real);

	if (TRACE(GOMP_sections_start_real))
	{
		Probe_OpenMP_Section_Entry();
		res = GOMP_sections_start_real (count);
		Probe_OpenMP_Section_Exit();
	}
	else if (GOMP_sections_start_real != NULL)
	{
		res = GOMP_sections_start_real (count);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_start exit: res=%u\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

unsigned GOMP_sections_next (void)
{
	unsigned res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_next enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_sections_next_real);
#endif

	RECHECK_INIT(GOMP_sections_next_real);

	if (TRACE(GOMP_sections_next_real))
	{
		Probe_OpenMP_Work_Entry();
		res = GOMP_sections_next_real();
		Probe_OpenMP_Work_Exit();
	}
	else if (GOMP_sections_next_real != NULL)
	{
		res = GOMP_sections_next_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_next: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_next exit: res=%u\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

void GOMP_sections_end (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_end enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_sections_end_real);
#endif

	RECHECK_INIT(GOMP_sections_end_real);

	if (TRACE(GOMP_sections_end_real))
	{
		Probe_OpenMP_Join_Wait_Entry();
		GOMP_sections_end_real();
		Probe_OpenMP_Join_Wait_Exit();
	}
	else if (GOMP_sections_end_real != NULL)
	{
		GOMP_sections_end_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_end: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_sections_end_nowait (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_end_nowait enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_sections_end_nowait_real);
#endif

	RECHECK_INIT(GOMP_sections_end_nowait_real);

	if (TRACE(GOMP_sections_end_nowait_real))
	{
		Probe_OpenMP_Join_NoWait_Entry();
		GOMP_sections_end_nowait_real();
		Probe_OpenMP_Join_NoWait_Exit();
	}
	else if (GOMP_sections_end_nowait_real != NULL)
	{
		GOMP_sections_end_nowait_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_end_nowait: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_sections_end_nowait exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

unsigned GOMP_single_start (void)
{
	unsigned res = 0;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_single_start enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_single_start_real);
#endif

	RECHECK_INIT(GOMP_single_start_real);

	if (TRACE(GOMP_single_start_real))
	{
		Probe_OpenMP_Single_Entry();
		res = GOMP_single_start_real ();
		Probe_OpenMP_Single_Exit();
	}
	else if (GOMP_single_start_real != NULL)
	{
		res = GOMP_single_start_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_single_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_single_start exit: res=%u\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

void GOMP_taskwait (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskwait enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_taskwait_real);
#endif

	RECHECK_INIT(GOMP_taskwait_real);

	if (TRACE(GOMP_taskwait_real))
	{
		Probe_OpenMP_Taskwait_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		Backend_Leave_Instrumentation ();
		GOMP_taskwait_real ();
		Backend_Enter_Instrumentation ();

		Probe_OpenMP_Taskwait_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else if (GOMP_taskwait_real != NULL)
	{
		GOMP_taskwait_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskwait: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskwait exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

/* As of 2025-02-28, the implementation of GOMP_taskyield in the GNU OpenMP runtime is empty:
 * https://github.com/gcc-mirror/gcc/blob/master/libgomp/task.c#L2164 
 * https://gcc.gnu.org/legacy-ml/gcc-patches/2011-08/msg00080.html 
 * The wrapper is included just to mark in the trace where a pragma taskyield appears in the user code.
 */
void GOMP_taskyield (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskyield enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_taskyield_real);
#endif

	RECHECK_INIT(GOMP_taskyield_real);

	if (TRACE(GOMP_taskyield_real))
	{
		Probe_OpenMP_Taskyield_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		GOMP_taskyield_real ();
		Probe_OpenMP_Taskyield_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else if (GOMP_taskyield_real != NULL)
	{
		GOMP_taskyield_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskyield: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskyield exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}


/********************************************/
/***** Added (or changed) in OpenMP 4.0 *****/
/********************************************/

void
GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads, unsigned int flags)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel enter: @=%p args=(%p %p %u %u)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_real, fn, data, num_threads, flags);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_real);

	if (TRACE(GOMP_parallel_real))
	{
		SAVE_PARALLEL_UF(fn);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		/*
		 * GOMP_parallel has an implicit join, so when we return, the helpers
		 * can be freed, so rather than using the static array of active
		 * helpers, we statically declare a private helper and pass it by
		 * reference.
		 * void *par_helper = __GOMP_new_helper(fn, data);
		 */
		struct parallel_helper_t par_helper;
		par_helper.fn = fn;
		par_helper.data = data;

		Probe_OpenMP_ParRegion_Entry();
		Extrae_OpenMP_EmitTaskStatistics();

		Backend_Leave_Instrumentation ();
		GOMP_parallel_real(callme_par, &par_helper, num_threads, flags);
		Backend_Enter_Instrumentation ();

		Probe_OpenMP_ParRegion_Exit();
		Extrae_OpenMP_EmitTaskStatistics();

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_real != NULL)
	{
		GOMP_parallel_real(fn, data, num_threads, flags);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_parallel exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_loop_static(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr, long chunk_size, unsigned flags)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_static enter: @=%p args=(%p %p %u %ld %ld %ld %ld %u)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_static_real, fn, data, num_threads, start, end, incr, chunk_size, flags);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_static_real);

	if (TRACE(GOMP_parallel_loop_static_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_static_real(callme_pardo, pardo_helper, num_threads, start, end, incr, chunk_size, flags);
		Backend_Enter_Instrumentation ();
		Probe_OpenMP_ParDO_Exit();

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_static_real != NULL)
	{
		GOMP_parallel_loop_static_real(fn, data, num_threads, start, end, incr, chunk_size, flags);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_static: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_static exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_loop_dynamic(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr, long chunk_size, unsigned flags)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_dynamic enter: @=%p args=(%p %p %u %ld %ld %ld %ld %u)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_dynamic_real, fn, data, num_threads, start, end, incr, chunk_size, flags);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_dynamic_real);

	if (TRACE(GOMP_parallel_loop_dynamic_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_dynamic_real(callme_pardo, pardo_helper, num_threads, start, end, incr, chunk_size, flags);
		Backend_Enter_Instrumentation ();
		Probe_OpenMP_ParDO_Exit();

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_dynamic_real != NULL)
	{
		GOMP_parallel_loop_dynamic_real(fn, data, num_threads, start, end, incr, chunk_size, flags);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_dynamic: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_parallel_loop_dynamic exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_loop_guided(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr, long chunk_size, unsigned flags)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_guided enter: @=%p args=(%p %p %u %ld %ld %ld %ld %u)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_guided_real, fn, data, num_threads, start, end, incr, chunk_size, flags);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_guided_real);

	if (TRACE(GOMP_parallel_loop_guided_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_guided_real(callme_pardo, pardo_helper, num_threads, start, end, incr, chunk_size, flags);
		Backend_Enter_Instrumentation ();
		Probe_OpenMP_ParDO_Exit();

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_guided_real != NULL)
	{
		GOMP_parallel_loop_guided_real(fn, data, num_threads, start, end, incr, chunk_size, flags);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_guided: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_guided exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_loop_runtime(void (*fn)(void *), void *data, unsigned num_threads, long start, long end, long incr, unsigned flags)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_runtime enter: @=%p args=(%p %p %u %ld %ld %ld %u)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_loop_runtime_real, fn, data, num_threads, start, end, incr, flags);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_loop_runtime_real);

	if (TRACE(GOMP_parallel_loop_runtime_real))
	{
		void *pardo_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParDO_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_loop_runtime_real(callme_pardo, pardo_helper, num_threads, start, end, incr, flags);
		Backend_Enter_Instrumentation ();
		Probe_OpenMP_ParDO_Exit();

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_loop_runtime_real != NULL)
	{
		GOMP_parallel_loop_runtime_real(fn, data, num_threads, start, end, incr, flags);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_loop_runtime: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_loop_runtime exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void
GOMP_parallel_sections(void (*fn) (void *), void *data, unsigned num_threads, unsigned count, unsigned flags)
{
	Backend_Enter_Instrumentation ();
	#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_sections enter: @=%p args=(%p %p %u %u %u)\n",
	    THREAD_LEVEL_VAR, GOMP_parallel_sections_real, fn, data, num_threads, count, flags);
#endif

	OMP_CLAUSE_NUM_THREADS_SAVE(original_num_threads);

	RECHECK_INIT(GOMP_parallel_sections_real);

	if (TRACE(GOMP_parallel_sections_real))
	{
		void *parsections_helper = __GOMP_new_helper(fn, data);

		/*
		 * Change number of threads if specified by parallel directive only if
		 * in a library not mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(num_threads);

		Probe_OpenMP_ParSections_Entry();
		Backend_Leave_Instrumentation ();
		GOMP_parallel_sections_real(callme_parsections, parsections_helper, num_threads, count, flags);
		Backend_Enter_Instrumentation ();
		Probe_OpenMP_ParSections_Exit();

		/* The master thread continues the execution and then calls parsections_helper->fn */
		if (THREADID == 0)
		{
			Probe_OpenMP_UF_Entry(fn);
		}

		/*
		 * Return to the original number of threads only if in a library not
		 * mixing runtimes.
		 */
		OMP_CLAUSE_NUM_THREADS_CHANGE(original_num_threads);
	}
	else if (GOMP_parallel_sections_real != NULL)
	{
		GOMP_parallel_sections_real(fn, data, num_threads, count, flags);
	}
	else
	{
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
		    "GOMP_parallel_sections: This function is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL
	    "GOMP_parallel_sections exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}


void GOMP_taskgroup_start (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskgroup_start enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_taskgroup_start_real);
#endif

	RECHECK_INIT(GOMP_taskgroup_start_real);

	if (TRACE(GOMP_taskgroup_start_real))
	{
		Probe_OpenMP_Taskgroup_start_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		GOMP_taskgroup_start_real ();
		Probe_OpenMP_Taskgroup_start_Exit();
	}
	else if (GOMP_taskgroup_start_real != NULL)
	{
		GOMP_taskgroup_start_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskgroup_start: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskgroup_start exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

void GOMP_taskgroup_end (void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskgroup_end enter: @=%p\n", THREAD_LEVEL_VAR, GOMP_taskgroup_end_real);
#endif

	RECHECK_INIT(GOMP_taskgroup_end_real);

	if (TRACE(GOMP_taskgroup_end_real))
	{
		Probe_OpenMP_Taskgroup_end_Entry();
		GOMP_taskgroup_end_real ();
		Probe_OpenMP_Taskgroup_end_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else if (GOMP_taskgroup_end_real != NULL)
	{
		GOMP_taskgroup_end_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskgroup_end: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskgroup_end exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target
 */
void GOMP_target(int device, void (*fn) (void *), const void *unused,
                 size_t mapnum, void **hostaddrs, size_t *sizes,
                 unsigned char *kinds)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_real);
#endif

	RECHECK_INIT(GOMP_target_real);

	if (TRACE(GOMP_target_real))
	{
		Probe_OpenMP_Target_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		GOMP_target_real(device, fn, unused, mapnum, hostaddrs, sizes, kinds);
		Probe_OpenMP_Target_Exit();
	} else if (GOMP_target_real != NULL)
	{
		GOMP_target_real(device, fn, unused, mapnum, hostaddrs, sizes, kinds);
	} else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked!  Exiting!!\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target_data
 */
void
GOMP_target_data(int device, const void *unused, size_t mapnum,
                 void **hostaddrs, size_t *sizes, unsigned char *kinds)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_data_real);
#endif

	RECHECK_INIT(GOMP_target_data_real);

	if (TRACE(GOMP_target_data_real))
	{
		Probe_OpenMP_Target_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		GOMP_target_data_real(device, unused, mapnum, hostaddrs, sizes, kinds);
		Probe_OpenMP_Target_Exit();
	} else if (GOMP_target_data_real != NULL)
	{
		GOMP_target_data_real(device, unused, mapnum, hostaddrs, sizes, kinds);
	} else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked!  Exiting ...\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target_end_data
 */
void
GOMP_target_end_data(void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_end_data_real);
#endif

	RECHECK_INIT(GOMP_target_end_data_real);

	if (TRACE(GOMP_target_end_data_real))
	{
		Probe_OpenMP_Target_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		GOMP_target_end_data_real();
		Probe_OpenMP_Target_Exit();
	} else if (GOMP_target_end_data_real != NULL)
	{
		GOMP_target_end_data_real();
	} else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked!  Exiting ...\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target_update
 */
void
GOMP_target_update(int device, const void *unused, size_t mapnum,
                   void **hostaddrs, size_t *sizes, unsigned char *kinds)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_update_real);
#endif

	RECHECK_INIT(GOMP_target_update_real);

	if (TRACE(GOMP_target_update_real))
	{
		Probe_OpenMP_Target_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		GOMP_target_update_real(device, unused, mapnum, hostaddrs, sizes, kinds);
		Probe_OpenMP_Target_Exit();
	} else if (GOMP_target_update_real != NULL)
	{
		GOMP_target_update_real(device, unused, mapnum, hostaddrs, sizes, kinds);
	} else {
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked!  Exiting ...\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}


/********************************************/
/***** Added (or changed) in OpenMP 4.5 *****/
/********************************************/

/*
 * The prototype of GOMP_task changed in 4.0 to add a new parameter 'depend',
 * and again in 4.5 to add 'priority'. We define this prototype as receiving 
 * varargs, and we check the runtime version to decide with how many parameters
 * we will make the call to the real function.
 */
void GOMP_task (void (*fn)(void *), void *data, void (*cpyfn)(void *, void *), long arg_size, long arg_align, int if_clause, unsigned flags, ...)
{
	void **depend = NULL;
	int priority = 0;
	va_list ap;

	va_start (ap, flags);

	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_task enter: @=%p args=(%p %p %p %ld %ld %d %u) (GOMP version: %s)\n", THREAD_LEVEL_VAR, GOMP_task_real, fn, data, cpyfn, arg_size, arg_align, if_clause, flags, __GOMP_version);
#endif

	RECHECK_INIT(GOMP_task_real);

	if (TRACE(GOMP_task_real))
	{
		Probe_OpenMP_Task_Entry (fn);
		Extrae_OpenMP_Notify_NewInstantiatedTask();

		/* 
		 * Helpers for GOMP_task don't use the array of active helpers, as we know 
		 * that we can free them right away after the task is executed.
		 */
		struct task_helper_t *task_helper = (struct task_helper_t *) xmalloc(sizeof(struct task_helper_t));
		task_helper->fn = fn;
		task_helper->data = data;

		if (cpyfn != NULL)
		{
			char *buf = xmalloc(sizeof(char) * (arg_size + arg_align - 1));
			char *arg = (char *) (((uintptr_t) buf + arg_align - 1)
			            & ~(uintptr_t) (arg_align - 1));
			cpyfn (arg, data);
			task_helper->data = arg;
			// Saved for deallocation purposes, arg is not valid since includes offset
			task_helper->buf = buf; 
		}
		else
		{
			char *buf = xmalloc(sizeof(char) * (arg_size + arg_align - 1));
			memcpy (buf, data, arg_size);
			task_helper->data = buf;
			// Saved for deallocation purposes, arg is not valid since includes offset
			task_helper->buf = buf;
		}

#if defined(HAVE__SYNC_FETCH_AND_ADD)
		task_helper->counter = __sync_fetch_and_add(&__GOMP_task_ctr, 1);
#else
		pthread_mutex_lock (&__GOMP_task_ctr_mtx);
		task_helper->counter = __GOMP_task_ctr++;
		pthread_mutex_unlock (&__GOMP_task_ctr_mtx);
#endif

		Extrae_OpenMP_TaskID (task_helper->counter, XTR_TASK_INSTANTIATION);

		if (strcmp(__GOMP_version, GOMP_API_3_1) == 0) {
			GOMP_task_real (callme_task, &task_helper, NULL, sizeof(task_helper), arg_align, if_clause, flags);
		} else if (strcmp(__GOMP_version, GOMP_API_4_0) == 0) {
			depend = va_arg(ap, void **);
			GOMP_task_real (callme_task, &task_helper, NULL, sizeof(task_helper), arg_align, if_clause, flags, depend);
		} else if (strcmp(__GOMP_version, GOMP_API_4_5) == 0) {
			depend = va_arg(ap, void **);
			priority = va_arg(ap, int);
			GOMP_task_real (callme_task, &task_helper, NULL, sizeof(task_helper), arg_align, if_clause, flags, depend, priority);
		}

		Probe_OpenMP_Task_Exit ();
	}
	else if (GOMP_task_real != NULL)
	{
		if (strcmp(__GOMP_version, GOMP_API_3_1) == 0) {
			GOMP_task_real (fn, data, cpyfn, arg_size, arg_align, if_clause, flags);
		} else if (strcmp(__GOMP_version, GOMP_API_4_0) == 0) {
			depend = va_arg(ap, void **);
			GOMP_task_real (fn, data, cpyfn, arg_size, arg_align, if_clause, flags, depend);
		} else if (strcmp(__GOMP_version, GOMP_API_4_5) == 0) {
			depend = va_arg(ap, void **);
			priority = va_arg(ap, int);
			GOMP_task_real (fn, data, cpyfn, arg_size, arg_align, if_clause, flags, depend, priority);
		}
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_task: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

	va_end(ap);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_task exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}


void GOMP_taskloop (void *fn, void *data, void *cpyfn, long arg_size, long arg_align, unsigned flags, unsigned long num_tasks, int priority, long start, long end, long step)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskloop enter: @=%p args=(%p %p %p %ld %ld %u %lu %d %ld %ld %ld)\n", THREAD_LEVEL_VAR, GOMP_taskloop_real, fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step);
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskloop: instrumentation is %s\n", THREAD_LEVEL_VAR, (getTrace_OpenMP_Taskloop() ? "enabled" : "disabled"));
#endif

	RECHECK_INIT(GOMP_taskloop_real);

	if (TRACE(GOMP_taskloop_real) && (getTrace_OpenMP_Taskloop()))
	{
		struct taskloop_helper_t taskloop_helper;
		long helper_size = sizeof(struct taskloop_helper_t);

		taskloop_helper.magicno = (void *)0xdeadbeef; // Magic number to locate the helper in callme_taskloop
		taskloop_helper.fn = fn;

		// Assign a unique id to this taskloop to track when it is scheduled and when executed 
#if defined(HAVE__SYNC_FETCH_AND_ADD)
		taskloop_helper.id = __sync_fetch_and_add(&__GOMP_taskloop_ctr, 1);
#else
		pthread_mutex_lock (&__GOMP_taskloop_ctr_mtx);
		taskloop_helper.id = __GOMP_taskloop_ctr++;
		pthread_mutex_unlock (&__GOMP_taskloop_ctr_mtx);
#endif

    	        // Append the helper to the end of data
		void *data_trailer = xmalloc(arg_size + helper_size);
		memcpy (data_trailer, data, arg_size);
		memcpy (data_trailer + arg_size, &taskloop_helper, helper_size);

		Probe_OpenMP_TaskLoop_Entry ();
		GOMP_taskloop_real(callme_taskloop, data_trailer, cpyfn, arg_size + helper_size, arg_align, flags, num_tasks, priority, start, end, step);
		Probe_OpenMP_TaskLoop_Exit ();
	}
	else if (GOMP_taskloop_real != NULL)
	{
		GOMP_taskloop_real (fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskloop: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}
	Backend_Leave_Instrumentation ();
}

void GOMP_taskloop_ull (void *fn, void *data, void *cpyfn, long arg_size, long arg_align, unsigned flags, unsigned long num_tasks, int priority, unsigned long long start, unsigned long long end, unsigned long long step)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskloop_ull enter: @=%p args=(%p %p %p %ld %ld %u %lu %d %llu %llu %llu)\n", THREAD_LEVEL_VAR, GOMP_taskloop_ull_real, fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step);
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskloop_ull: instrumentation is %s\n", THREAD_LEVEL_VAR, (getTrace_OpenMP_Taskloop() ? "enabled" : "disabled"));
#endif

	RECHECK_INIT(GOMP_taskloop_ull_real);
	if (TRACE(GOMP_taskloop_ull_real) && (getTrace_OpenMP_Taskloop()))
	{
		struct taskloop_helper_t taskloop_helper;
		long helper_size = sizeof(struct taskloop_helper_t);

		taskloop_helper.magicno = (void *)0xdeadbeef; // Magic number to locate the helper in callme_taskloop
		taskloop_helper.fn = fn;

		// Assign a unique id to this taskloop to track when it is scheduled and when executed 
#if defined(HAVE__SYNC_FETCH_AND_ADD)
		taskloop_helper.id = __sync_fetch_and_add(&__GOMP_taskloop_ctr, 1);
#else
		pthread_mutex_lock (&__GOMP_taskloop_ctr_mtx);
		taskloop_helper.id = __GOMP_taskloop_ctr++;
		pthread_mutex_unlock (&__GOMP_taskloop_ctr_mtx);
#endif
    	        // Append the helper to the end of data
		void *data_trailer = xmalloc(arg_size + helper_size);
		memcpy (data_trailer, data, arg_size);
		memcpy (data_trailer + arg_size, &taskloop_helper, helper_size);

		Probe_OpenMP_TaskLoop_Entry ();
		GOMP_taskloop_ull_real(callme_taskloop, data_trailer, cpyfn, arg_size + helper_size, arg_align, flags, num_tasks, priority, start, end, step);
		Probe_OpenMP_TaskLoop_Exit ();
	}
	else if (GOMP_taskloop_ull_real != NULL)
	{
		GOMP_taskloop_ull_real(fn, data, cpyfn, arg_size, arg_align, flags, num_tasks, priority, start, end, step);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskloop_ull: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_taskloop_ull exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

int GOMP_loop_doacross_static_start (unsigned ncounts, long *counts, long chunk_size, long *istart, long *iend)
{
	int res;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_static_start enter: @=%p args=(%u %p %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_doacross_static_start_real, ncounts, counts, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_doacross_static_start_real);

	/* Save ncounts in the thread TLS */
	SAVE_DOACROSS_NCOUNTS(ncounts);

	if (TRACE(GOMP_loop_doacross_static_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_doacross_static_start_real (ncounts, counts, chunk_size, istart, iend);
	}
	else if (GOMP_loop_doacross_static_start_real != NULL)
	{
		res = GOMP_loop_doacross_static_start_real (ncounts, counts, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_static_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_static_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_doacross_dynamic_start (unsigned ncounts, long *counts, long chunk_size, long *istart, long *iend)
{
	int res;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_dynamic_start enter: @=%p args=(%u %p %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_doacross_dynamic_start_real, ncounts, counts, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_doacross_dynamic_start_real);

	/* Save ncounts in the thread TLS */
	SAVE_DOACROSS_NCOUNTS(ncounts);

	if (TRACE(GOMP_loop_doacross_dynamic_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_doacross_dynamic_start_real (ncounts, counts, chunk_size, istart, iend);
	}
	else if (GOMP_loop_doacross_dynamic_start_real != NULL)
	{
		res = GOMP_loop_doacross_dynamic_start_real (ncounts, counts, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_dynamic_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_dynamic_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_doacross_guided_start (unsigned ncounts, long *counts, long chunk_size, long *istart, long *iend)
{
	int res;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_guided_start enter: @=%p args=(%u %p %ld %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_doacross_guided_start_real, ncounts, counts, chunk_size, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_doacross_guided_start_real);

	/* Save ncounts in the thread TLS */
	SAVE_DOACROSS_NCOUNTS(ncounts);

	if (TRACE(GOMP_loop_doacross_guided_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_doacross_guided_start_real (ncounts, counts, chunk_size, istart, iend);
	}
	else if (GOMP_loop_doacross_guided_start_real != NULL)
	{
		res = GOMP_loop_doacross_guided_start_real (ncounts, counts, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_guided_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_guided_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

int GOMP_loop_doacross_runtime_start (unsigned ncounts, long *counts, long *istart, long *iend)
{
	int res;
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_runtime_start enter: @=%p args=(%u %p %p %p)\n", THREAD_LEVEL_VAR, GOMP_loop_doacross_runtime_start_real, ncounts, counts, istart, iend);
#endif

	RECHECK_INIT(GOMP_loop_doacross_runtime_start_real);

	/* Save ncounts in the thread TLS */
	SAVE_DOACROSS_NCOUNTS(ncounts);

	if (TRACE(GOMP_loop_doacross_runtime_start_real))
	{
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_doacross_runtime_start_real (ncounts, counts, istart, iend);
	}
	else if (GOMP_loop_doacross_runtime_start_real != NULL)
	{
		res = GOMP_loop_doacross_runtime_start_real (ncounts, counts, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_runtime_start_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_loop_doacross_runtime_start exit: res=%d\n", THREAD_LEVEL_VAR, res);
#endif
	Backend_Leave_Instrumentation ();

	return res;
}

void GOMP_doacross_post (long *counts)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_doacross_post enter: @=%p args=(%p)\n", THREAD_LEVEL_VAR, GOMP_doacross_post_real, counts);
#endif

	RECHECK_INIT(GOMP_doacross_post_real);

	if (TRACE(GOMP_doacross_post_real))
	{
		Probe_OpenMP_Ordered_Post_Entry();
		GOMP_doacross_post_real (counts);
		Probe_OpenMP_Ordered_Post_Exit();
	}
	else if (GOMP_doacross_post_real != NULL)
	{
		GOMP_doacross_post_real (counts);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_doacross_post: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_doacross_post exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

/*
 * The body of GOMP_doacross_wait needs a switch with as many cases as possible
 * values of 'ncounts'. These cases are generated dynamically with the script
 * 'genstubs-libgomp.sh' up to a maximum of DOACROSS_MAX_NESTING, defined in 
 * that script.
 */
void GOMP_doacross_wait (long first, ...)
{
	unsigned i = 0;
	long args[MAX_DOACROSS_ARGS];
	va_list ap;

	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_doacross_wait enter: @=%p args=(%ld)\n", THREAD_LEVEL_VAR, GOMP_doacross_wait_real, first);
#endif

	RECHECK_INIT(GOMP_doacross_wait_real);

	/* Retrieve ncounts from the thread TLS */
	unsigned ncounts = RETRIEVE_DOACROSS_NCOUNTS();

	if (TRACE(GOMP_doacross_wait_real))
	{
		va_start(ap, first);
		for (i = 0; i < ncounts; i++)                                               
		{                                                                           
			args[i] = va_arg(ap, long);                                               
		} 
		va_end(ap);	

		Probe_OpenMP_Ordered_Wait_Entry();

		switch (ncounts)
		{
			#include "gnu-libgomp-intermediate/libgomp-doacross-intermediate-switch.c"
			default:
				fprintf (stderr, PACKAGE_NAME": ERROR! Unhandled GOMP_doacross_wait call with %d arguments! Re-run the script 'genstubs-libgomp.sh' increasing the value of DOACROSS_MAX_NESTING. Quitting!\n"
						PACKAGE_NAME":        Check Extrae FAQ Webpage (https://tools.bsc.es/sites/default/files/documentation/html/extrae/FAQ.html#execution-faq) for more details", ncounts);
				exit(-1);
				break;
		}	

		Probe_OpenMP_Ordered_Wait_Exit();
	}
	else if (GOMP_doacross_wait_real != NULL)
	{
		va_start(ap, first);
		for (i = 0; i < ncounts; i++)
		{
			args[i] = va_arg(ap, long);
		}
		va_end(ap);

		switch (ncounts)
		{
			#include "gnu-libgomp-intermediate/libgomp-doacross-intermediate-switch.c"
			default:
				fprintf (stderr, PACKAGE_NAME": ERROR! Unhandled GOMP_doacross_wait call with %d arguments! Re-run the script 'genstubs-libgomp.sh' increasing the value of DOACROSS_MAX_NESTING. Quitting!\n"
						PACKAGE_NAME":        Check Extrae FAQ Webpage (https://tools.bsc.es/sites/default/files/documentation/html/extrae/FAQ.html#execution-faq) for more details", ncounts);
				exit(-1);
				break;
		}
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "ERROR! GOMP_doacross_wait_real: This function is not hooked! Exiting!!\n", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "GOMP_doacross_wait exit\n", THREAD_LEVEL_VAR);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target_ext
 */
void GOMP_target_ext(int device, void (*fn) (void *), size_t mapnum,
                     void **hostaddrs, size_t *sizes, unsigned short *kinds,
                     unsigned int flags, void **depend, void **args)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_ext_real);
#endif

	RECHECK_INIT(GOMP_target_ext_real);

	if (TRACE(GOMP_target_ext_real))
	{
		Probe_OpenMP_Target_Entry();
		GOMP_target_ext_real(device, fn, mapnum, hostaddrs, sizes, kinds, flags, depend, args);
		Probe_OpenMP_Target_Exit();
	}
	else if (GOMP_target_ext_real != NULL)
	{
		GOMP_target_ext_real(device, fn, mapnum, hostaddrs, sizes, kinds, flags, depend, args);
	}
	else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target_data_ext
 */
void
GOMP_target_data_ext(int device, size_t mapnum, void **hostaddrs,
                     size_t *sizes, unsigned short *kinds)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_data_ext_real);
#endif

	RECHECK_INIT(GOMP_target_data_ext_real);

	if (TRACE(GOMP_target_data_ext_real))
	{
		Probe_OpenMP_Target_Entry();
		GOMP_target_data_ext_real(device, mapnum, hostaddrs, sizes, kinds);
		Probe_OpenMP_Target_Exit();
	}
	else if (GOMP_target_data_ext_real != NULL)
	{
		GOMP_target_data_ext_real(device, mapnum, hostaddrs, sizes, kinds);
	}
	else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target_end_data_ext
 */
void
GOMP_target_end_data_ext(void)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_end_data_ext_real);
#endif

	RECHECK_INIT(GOMP_target_end_data_ext_real);

	if (TRACE(GOMP_target_end_data_ext_real))
	{
		Probe_OpenMP_Target_Entry();
		GOMP_target_end_data_ext_real();
		Probe_OpenMP_Target_Exit();
	}
	else if (GOMP_target_end_data_ext_real != NULL)
	{
		GOMP_target_end_data_ext_real();
	}
	else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
}

/**
 * GOMP_target_update_ext
 */
void
GOMP_target_update_ext(int device, size_t mapnum, void **hostaddrs,
                       size_t *sizes, unsigned short *kinds,
                       unsigned int flags, void **depend)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_update_ext_real);
#endif

	RECHECK_INIT(GOMP_target_update_ext_real);

	if (TRACE(GOMP_target_update_ext_real))
	{
		Probe_OpenMP_Target_Entry();
		GOMP_target_update_ext_real(device, mapnum, hostaddrs, sizes, kinds, flags, depend);
		Probe_OpenMP_Target_Exit();
	}
	else if (GOMP_target_update_ext_real != NULL)
	{
		GOMP_target_update_ext_real(device, mapnum, hostaddrs, sizes, kinds, flags, depend);
	}
	else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}

/**
 * GOMP_target_enter_exit_data
 */
void
GOMP_target_enter_exit_data(int device, size_t mapnum, void **hostaddrs,
                            size_t *sizes, unsigned short *kinds,
                            unsigned int flags, void **depend)
{
	Backend_Enter_Instrumentation ();
#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s enter: @=%p\n",
	    THREAD_LEVEL_VAR, __func__, GOMP_target_enter_exit_data_real);
#endif

	RECHECK_INIT(GOMP_target_enter_exit_data_real);

	if (TRACE(GOMP_target_enter_exit_data_real))
	{
		Probe_OpenMP_Target_Entry();
		GOMP_target_enter_exit_data_real(device, mapnum, hostaddrs, sizes, kinds, flags, depend);
		Probe_OpenMP_Target_Exit();
	}
	else if (GOMP_target_enter_exit_data_real != NULL)
	{
		GOMP_target_enter_exit_data_real(device, mapnum, hostaddrs, sizes, kinds, flags, depend);
	}
	else
	{
		fprintf(stderr,
		    PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s is not hooked! Exiting!!\n",
		    THREAD_LEVEL_VAR, __func__);
		exit(-1);
	}

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "%s exit\n",
	    THREAD_LEVEL_VAR, __func__);
#endif
	Backend_Leave_Instrumentation ();
}


/******************************************************************************\
 *                                                                            *
 *                             INITIALIZATIONS                                *
 *                                                                            *
\******************************************************************************/

/**
 * gnu_libgomp_get_hook_points
 *
 * Find the real implementation of the functions. We use dlsym to find the next
 * definition of the different symbols of the OpenMP runtime (i.e. skip our
 * wrapper, find the real one). 
 *
 * @param rank The current process ID (not used).
 *
 * @return 1 if any hook was found; 0 otherwise.
 */
static int gnu_libgomp_get_hook_points (int rank)
{
	int count = 0;

	UNREFERENCED_PARAMETER(rank);

	// Detect the OpenMP version supported by the runtime
	if ((__GOMP_version = getenv("EXTRAE___GOMP_version")) != NULL) {
		if ((strcmp(__GOMP_version, GOMP_API_4_5) != 0) &&
		    (strcmp(__GOMP_version, GOMP_API_4_0) != 0) &&
		    (strcmp(__GOMP_version, GOMP_API_3_1) != 0)) {
			fprintf(stderr, PACKAGE_NAME": ERROR! Unsupported GOMP version (%s). Valid versions are: 3.1, 4.0 and 4.5. Exiting ...\n", __GOMP_version);
			exit (-1);
		}
	} else if (dlsym(RTLD_NEXT, "GOMP_taskloop") != NULL) {
		__GOMP_version = GOMP_API_4_5;
	} else if (dlsym(RTLD_NEXT, "GOMP_taskgroup_start") != NULL) {
		__GOMP_version = GOMP_API_4_0;
	} else {
		__GOMP_version = GOMP_API_3_1;
	}

	if (TASKID == 0)
	{
		fprintf (stdout, PACKAGE_NAME": Detected GOMP version is %s\n", __GOMP_version);
	}

  /**********************/
  /***** OpenMP 3.1 *****/
  /**********************/

	/* Obtain @ for GOMP_atomic_start */
	GOMP_atomic_start_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_atomic_start");
	INC_IF_NOT_NULL(GOMP_atomic_start_real,count);

	/* Obtain @ for GOMP_atomic_end */
	GOMP_atomic_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_atomic_end");
	INC_IF_NOT_NULL(GOMP_atomic_end_real,count);

	/* Obtain @ for GOMP_barrier */
	GOMP_barrier_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_barrier");
	INC_IF_NOT_NULL(GOMP_barrier_real,count);

	/* Obtain @ for GOMP_critical_enter */
	GOMP_critical_start_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_critical_start");
	INC_IF_NOT_NULL(GOMP_critical_start_real,count);

	/* Obtain @ for GOMP_critical_end */
	GOMP_critical_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_critical_end");
	INC_IF_NOT_NULL(GOMP_critical_end_real,count);

	/* Obtain @ for GOMP_critical_name_start */
	GOMP_critical_name_start_real =
		(void(*)(void**)) dlsym (RTLD_NEXT, "GOMP_critical_name_start");
	INC_IF_NOT_NULL(GOMP_critical_name_start_real,count);

	/* Obtain @ for GOMP_critical_name_end */
	GOMP_critical_name_end_real =
		(void(*)(void**)) dlsym (RTLD_NEXT, "GOMP_critical_name_end");
	INC_IF_NOT_NULL(GOMP_critical_name_end_real,count);

	/* Obtain @ for GOMP_loop_static_start */
	GOMP_loop_static_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_static_start");
	INC_IF_NOT_NULL(GOMP_loop_static_start_real,count);

	/* Obtain @ for GOMP_loop_dynamic_start */
	GOMP_loop_dynamic_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_dynamic_start");
	INC_IF_NOT_NULL(GOMP_loop_dynamic_start_real,count);

	/* Obtain @ for GOMP_loop_guided_start */
	GOMP_loop_guided_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_guided_start");
	INC_IF_NOT_NULL(GOMP_loop_guided_start_real,count);

	/* Obtain @ for GOMP_loop_runtime_start */
	GOMP_loop_runtime_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_runtime_start");
	INC_IF_NOT_NULL(GOMP_loop_runtime_start_real,count);

	/* Obtain @ for GOMP_loop_static_next */
	GOMP_loop_static_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_static_next");
	INC_IF_NOT_NULL(GOMP_loop_static_next_real,count);

	/* Obtain @ for GOMP_loop_dynamic_next */
	GOMP_loop_dynamic_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_dynamic_next");
	INC_IF_NOT_NULL(GOMP_loop_dynamic_next_real,count);

	/* Obtain @ for GOMP_loop_guided_next */
	GOMP_loop_guided_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_guided_next");
	INC_IF_NOT_NULL(GOMP_loop_guided_next_real,count);

	/* Obtain @ for GOMP_loop_runtime_next */
	GOMP_loop_runtime_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_runtime_next");
	INC_IF_NOT_NULL(GOMP_loop_runtime_next_real,count);

	/* Obtain @ for GOMP_loop_ordered_static_start */
	GOMP_loop_ordered_static_start_real =
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_static_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_static_start_real, count);

	/* Obtain @ for GOMP_loop_ordered_dynamic_start */
	GOMP_loop_ordered_dynamic_start_real =
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_dynamic_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_dynamic_start_real, count);

	/* Obtain @ for GOMP_loop_ordered_guided_start */
	GOMP_loop_ordered_guided_start_real =
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_guided_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_guided_start_real, count);

	/* Obtain @ for GOMP_loop_ordered_runtime_start */
	GOMP_loop_ordered_runtime_start_real =
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_runtime_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_runtime_start_real, count);

	/* Obtain @ for GOMP_loop_ordered_static_next */
	GOMP_loop_ordered_static_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_static_next");
	INC_IF_NOT_NULL(GOMP_loop_ordered_static_next_real,count);

	/* Obtain @ for GOMP_loop_ordered_dynamic_next */
	GOMP_loop_ordered_dynamic_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_dynamic_next");
	INC_IF_NOT_NULL(GOMP_loop_ordered_dynamic_next_real,count);

	/* Obtain @ for GOMP_loop_ordered_guided_next */
	GOMP_loop_ordered_guided_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_guided_next");
	INC_IF_NOT_NULL(GOMP_loop_ordered_guided_next_real,count);

	/* Obtain @ for GOMP_loop_runtime_next */
	GOMP_loop_ordered_runtime_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_runtime_next");
	INC_IF_NOT_NULL(GOMP_loop_ordered_runtime_next_real,count);

	/* Obtain @ for GOMP_parallel_loop_static_start */
	GOMP_parallel_loop_static_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_static_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_static_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_dynamic_start */
	GOMP_parallel_loop_dynamic_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_dynamic_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_dynamic_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_guided_start */
	GOMP_parallel_loop_guided_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_guided_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_guided_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_runtime_start */
	GOMP_parallel_loop_runtime_start_real =
		(void(*)(void*,void*,unsigned, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_runtime_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_runtime_start_real,count);

	/* Obtain @ for GOMP_loop_end */
	GOMP_loop_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_loop_end");
	INC_IF_NOT_NULL(GOMP_loop_end_real,count);

	/* Obtain @ for GOMP_loop_end_nowait */
	GOMP_loop_end_nowait_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_loop_end_nowait");
	INC_IF_NOT_NULL(GOMP_loop_end_nowait_real,count);

	/* Obtain @ for GOMP_ordered_start */
	GOMP_ordered_start_real = (void(*)(void)) dlsym (RTLD_NEXT, "GOMP_ordered_start");
	INC_IF_NOT_NULL(GOMP_ordered_start_real,count);

	/* Obtain @ for GOMP_ordered_end */
	GOMP_ordered_end_real = (void(*)(void)) dlsym (RTLD_NEXT, "GOMP_ordered_end");
	INC_IF_NOT_NULL(GOMP_ordered_end_real,count);

	/* Obtain @ for GOMP_parallel_start */
	GOMP_parallel_start_real =
		(void(*)(void*,void*,unsigned)) dlsym (RTLD_NEXT, "GOMP_parallel_start");
	INC_IF_NOT_NULL(GOMP_parallel_start_real,count);

	/* Obtain @ for GOMP_parallel_end */
	GOMP_parallel_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_parallel_end");
	INC_IF_NOT_NULL(GOMP_parallel_end_real,count);

	/* Obtain @ for GOMP_parallel_sections_start */
	GOMP_parallel_sections_start_real = 
		(void(*)(void*,void*,unsigned,unsigned)) dlsym (RTLD_NEXT, "GOMP_parallel_sections_start");
	INC_IF_NOT_NULL(GOMP_parallel_sections_start_real,count);

	/* Obtain @ for GOMP_sections_start */
	GOMP_sections_start_real =
		(unsigned(*)(unsigned)) dlsym (RTLD_NEXT, "GOMP_sections_start");
	INC_IF_NOT_NULL(GOMP_sections_start_real,count);

	/* Obtain @ for GOMP_sections_next */
	GOMP_sections_next_real =
		(unsigned(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_next");
	INC_IF_NOT_NULL(GOMP_sections_next_real,count);

	/* Obtain @ for GOMP_sections_end */
	GOMP_sections_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_end");
	INC_IF_NOT_NULL(GOMP_sections_end_real,count);

	/* Obtain @ for GOMP_sections_end_nowait */
	GOMP_sections_end_nowait_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_end_nowait");
	INC_IF_NOT_NULL(GOMP_sections_end_nowait_real,count);

	/* Obtain @ for GOMP_single_start */
	GOMP_single_start_real =
		(unsigned(*)(void)) dlsym (RTLD_NEXT, "GOMP_single_start");
	INC_IF_NOT_NULL(GOMP_single_start_real,count);

	/* Obtain @ for GOMP_taskwait */
	GOMP_taskwait_real = (void(*)(void)) dlsym (RTLD_NEXT, "GOMP_taskwait");
	INC_IF_NOT_NULL(GOMP_taskwait_real,count);

	/* Obtain @ for GOMP_taskyield */
	GOMP_taskyield_real = (void(*)(void)) dlsym (RTLD_NEXT, "GOMP_taskyield");
	INC_IF_NOT_NULL(GOMP_taskyield_real,count);

  /**********************/
  /***** OpenMP 4.0 *****/
  /**********************/

	/* Obtain @ for GOMP_parallel */
	GOMP_parallel_real =
		(void(*)(void*,void*,unsigned,unsigned int)) dlsym(RTLD_NEXT, "GOMP_parallel");
	INC_IF_NOT_NULL(GOMP_parallel_real,count);

	/* Obtain @ for GOMP_parallel_loop_static */
	GOMP_parallel_loop_static_real =
	    (void(*)(void*, void*, unsigned, long, long, long, long, unsigned))dlsym (RTLD_NEXT, "GOMP_parallel_loop_static");
	INC_IF_NOT_NULL(GOMP_parallel_loop_static_real,count);

	/* Obtain @ for GOMP_parallel_loop_dynamic */
	GOMP_parallel_loop_dynamic_real =
	    (void(*)(void*, void*, unsigned, long, long, long, long, unsigned))dlsym (RTLD_NEXT, "GOMP_parallel_loop_dynamic");
	INC_IF_NOT_NULL(GOMP_parallel_loop_dynamic_real,count);

	/* Obtain @ for GOMP_parallel_loop_guided */
	GOMP_parallel_loop_guided_real =
	    (void(*)(void*, void*, unsigned, long, long, long, long, unsigned))dlsym (RTLD_NEXT, "GOMP_parallel_loop_guided");
	INC_IF_NOT_NULL(GOMP_parallel_loop_guided_real,count);

	/* Obtain @ for GOMP_parallel_loop_runtime */
	GOMP_parallel_loop_runtime_real =
	    (void(*)(void*, void*, unsigned, long, long, long, unsigned)) dlsym(RTLD_NEXT, "GOMP_parallel_loop_runtime");
	INC_IF_NOT_NULL(GOMP_parallel_loop_runtime_real,count);

	/* Obtain @ for GOMP_parallel_sections */
	GOMP_parallel_sections_real =
	    (void(*)(void *, void *, unsigned, unsigned, unsigned)) dlsym(RTLD_NEXT, "GOMP_parallel_sections");
	INC_IF_NOT_NULL(GOMP_parallel_sections_real,count);

	/* Obtain @ for GOMP_taskgroup_start */
	GOMP_taskgroup_start_real = (void(*)(void)) dlsym(RTLD_NEXT, "GOMP_taskgroup_start");
	INC_IF_NOT_NULL(GOMP_taskgroup_start_real,count);

	/* Obtain @ for GOMP_taskgroup_end */
	GOMP_taskgroup_end_real = (void(*)(void)) dlsym(RTLD_NEXT, "GOMP_taskgroup_end");
	INC_IF_NOT_NULL(GOMP_taskgroup_end_real,count);

	/* Obtain @ for GOMP_target */
	GOMP_target_real = (void(*)(int, void *, const void *, size_t, void **, size_t *, unsigned char *)) dlsym (RTLD_NEXT, "GOMP_target");
	INC_IF_NOT_NULL(GOMP_target_real,count);

	/* Obtain @ for GOMP_target_data */
	GOMP_target_data_real = (void(*)(int, const void *, size_t, void **, size_t *, unsigned char *)) dlsym (RTLD_NEXT, "GOMP_target_data");
	INC_IF_NOT_NULL(GOMP_target_data_real,count);

	/* Obtain @ for GOMP_target_update */
	GOMP_target_update_real = (void(*)(int, const void *, size_t, void **, size_t *, unsigned char *)) dlsym (RTLD_NEXT, "GOMP_target_update");
	INC_IF_NOT_NULL(GOMP_target_update_real,count);

  /**********************/
  /***** OpenMP 4.5 *****/
  /**********************/

	/* Obtain @ for GOMP_task */
	GOMP_task_real =
		(void(*)(void*,void*,void*,long,long,int,unsigned,...)) dlsym (RTLD_NEXT, "GOMP_task");
	INC_IF_NOT_NULL(GOMP_task_real,count);

	/* Obtain @ for GOMP_taskloop */
	GOMP_taskloop_real =
		(void(*)(void*,void*,void*,long,long,unsigned,unsigned long,int,long,long,long)) dlsym (RTLD_NEXT, "GOMP_taskloop");
	INC_IF_NOT_NULL(GOMP_taskloop_real,count);

	/* Obtain @ for GOMP_taskloop_ull */
	GOMP_taskloop_ull_real =
		(void(*)(void*,void*,void*,long,long,unsigned,unsigned long,int,unsigned long long,unsigned long long,unsigned long long)) dlsym (RTLD_NEXT, "GOMP_taskloop_ull");
	INC_IF_NOT_NULL(GOMP_taskloop_ull_real,count);

	/* Obtain @ for GOMP_loop_doacross_static_start */
	GOMP_loop_doacross_static_start_real = (int(*)(unsigned, long *, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_doacross_static_start");
	INC_IF_NOT_NULL(GOMP_loop_doacross_static_start_real,count);

	/* Obtain @ for GOMP_loop_doacross_dynamic_start */
	GOMP_loop_doacross_dynamic_start_real = (int(*)(unsigned, long *, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_doacross_dynamic_start");
	INC_IF_NOT_NULL(GOMP_loop_doacross_dynamic_start_real,count);

	/* Obtain @ for GOMP_loop_doacross_guided_start */
	GOMP_loop_doacross_guided_start_real = (int(*)(unsigned, long *, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_doacross_guided_start");
	INC_IF_NOT_NULL(GOMP_loop_doacross_guided_start_real,count);

	/* Obtain @ for GOMP_loop_doacross_runtime_start */
	GOMP_loop_doacross_runtime_start_real = (int(*)(unsigned, long *, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_doacross_runtime_start");
	INC_IF_NOT_NULL(GOMP_loop_doacross_runtime_start_real,count);

	/* Obtain @ for GOMP_doacross_post */
	GOMP_doacross_post_real = (void(*)(long *)) dlsym (RTLD_NEXT, "GOMP_doacross_post");
	INC_IF_NOT_NULL(GOMP_doacross_post_real,count);

	/* Obtain @ for GOMP_doacross_wait */
	GOMP_doacross_wait_real = (void(*)(long, ...)) dlsym (RTLD_NEXT, "GOMP_doacross_wait");
	INC_IF_NOT_NULL(GOMP_doacross_wait_real,count);

	/* Obtain @ for GOMP_target_ext */
	GOMP_target_ext_real = (void(*)(int, void *, size_t, void **, size_t *, unsigned short *, unsigned int, void **, void **)) dlsym (RTLD_NEXT, "GOMP_target_ext");
	INC_IF_NOT_NULL(GOMP_target_ext_real,count);

	/* Obtain @ for GOMP_target_data_ext */
	GOMP_target_data_ext_real = (void(*)(int, size_t, void **, size_t *, unsigned short *)) dlsym (RTLD_NEXT, "GOMP_target_data_ext");
	INC_IF_NOT_NULL(GOMP_target_data_ext_real,count);

	/* Obtain @ for GOMP_target_update_ext */
	GOMP_target_update_ext_real = (void(*)(int, size_t, void **, size_t *, unsigned short *, unsigned int, void **)) dlsym (RTLD_NEXT, "GOMP_target_update_ext");
	INC_IF_NOT_NULL(GOMP_target_update_ext_real,count);

	/* Any hook point? */
	return (count > 0);
}

/**
 * _extrae_gnu_libgomp_init
 *
 * Initializes the instrumentation module for GNU libgomp.
 *
 * @param rank The current process ID (not used).
 */
int _extrae_gnu_libgomp_init (int rank)
{
	preallocate_GOMP_helpers();

	return gnu_libgomp_get_hook_points (rank);
}

#endif /* PIC */

