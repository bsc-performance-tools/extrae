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
# undef __USE_GNU
#endif
#ifdef HAVE_STDARG_H
# include <stdarg.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif

#include "intel-kmpc-11.h"
#include "wrapper.h"
#include "omp-common.h"
#include "omp-probe.h"
#include "omp-events.h"
#include "intel-kmpc-11-intermediate/intel-kmpc-11-intermediate.h"
#include "intel-kmpc-11-intermediate/intel-kmpc-11-taskloop-helpers.h"

//#define DEBUG

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
		_extrae_intel_kmpc_init(TASKID);                                   \
  }                                                                    \
}

#if defined(PIC)

static int intel_kmpc_get_hook_points (int rank);

static void (*ompc_set_num_threads_real)(int) = NULL;

static void (*__kmpc_barrier_real)(void*,int) = NULL;

static void (*__kmpc_critical_real)(void*,int,void*) = NULL;
static void (*__kmpc_end_critical_real)(void*,int,void*) = NULL;

static void (*__kmpc_dispatch_init_4_real)(void*,int,int,int,int,int,int) = NULL;
static void (*__kmpc_dispatch_init_8_real)(void*,int,int,long long,long long,long long,long long) = NULL;
static int (*__kmpc_dispatch_next_4_real)(void*,int,int*,int*,int*,int*) = NULL;
static int (*__kmpc_dispatch_next_8_real)(void*,int,int*,long long *,long long *, long long *) = NULL;
static void (*__kmpc_dispatch_fini_4_real)(void*,int) = NULL;
static void (*__kmpc_dispatch_fini_8_real)(void*,int) = NULL; 

void (*__kmpc_fork_call_real)(void*,int,void*,...) = NULL;

static int (*__kmpc_single_real)(void*,int) = NULL;
static void (*__kmpc_end_single_real)(void*,int) = NULL;

static void* (*__kmpc_omp_task_alloc_real)(void*,int,int,size_t,size_t,void*) = NULL;
static void (*__kmpc_omp_task_begin_if0_real)(void*,int,void*) = NULL;
static void (*__kmpc_omp_task_complete_if0_real)(void*,int,void*) = NULL;
static int (*__kmpc_omp_taskwait_real)(void*,int) = NULL;

static void (*__kmpc_taskloop_real)(void*,int,void*,int,void*,void*,long,int,int,long,void*) = NULL;


/******************************************************************************\
 *                                                                            *
 *                                  HELPERS                                   *
 *                                                                            *
 ******************************************************************************/

/*
 * The following helper structures are used to wrap the runtime's tasks with
 * wrappers to emit instrumentation. We store a list of tuples (pairs) of
 * real_task <-> wrap_task, the runtime is told to execute the wrap_task
 * and from the wrap_task we recover the real_task.
 */

struct helper__kmpc_task_t
{
	void *wrap_task;
	void *real_task;
};

struct helper_list__kmpc_task_t
{
	struct helper__kmpc_task_t *list;
	int count;
	int max_helpers;
};

/*
 * hl__kmpc_task contains a list of all active data helpers. The length
 * of the list can be increased from DEFAULT_OPENMP_HELPERS to the value set
 * by the EXTRAE_OPENMP_HELPERS environment variable.
 */
static pthread_mutex_t hl__kmpc_task_mtx = PTHREAD_MUTEX_INITIALIZER;
static struct helper_list__kmpc_task_t *hl__kmpc_task = NULL;

/*
 * Taskloop support relies partially on task support, but needs extra 
 * structures. See comments of __kmpc_taskloop function.
 */
struct helper_list__kmpc_taskloop_t
{
  void *real_task_map_by_helper[MAX_TASKLOOP_HELPERS];
	int next_id;
};

/*
 * hl__kmpc_taskloop contains a map of helper_id => real_task 
 */
static pthread_mutex_t hl__kmpc_taskloop_mtx = PTHREAD_MUTEX_INITIALIZER;
static struct helper_list__kmpc_taskloop_t *hl__kmpc_taskloop = NULL;


/**
 * preallocate_kmpc_helpers
 *
 * Allocates the helper structures for task and taskloop substitutions.
 */
static void preallocate_kmpc_helpers()
{
	int i = 0, num_helpers = 0;
	char *env_helpers = NULL;

	pthread_mutex_lock(&hl__kmpc_task_mtx);

	if (hl__kmpc_task == NULL)
	{
    hl__kmpc_task = (struct helper_list__kmpc_task_t *)malloc(sizeof(struct helper_list__kmpc_task_t));
		if (hl__kmpc_task == NULL)
		{
			fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "preallocate_kmpc_helpers: ERROR! Invalid initialization of 'hl__kmpc_task'\n ", THREAD_LEVEL_VAR);
			exit(-1);
		}

		/*                                                                          
     * If the environment variable ENV_VAR_EXTRAE_OPENMP_HELPERS is defined, this
     * will be the size of the list. Otherwise, DEFAULT_OPENMP_HELPERS is used.
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
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "preallocate_kmpc_helpers: Allocating %d data helpers\n ", THREAD_LEVEL_VAR, num_helpers);
#endif 

		hl__kmpc_task->count = 0;
		hl__kmpc_task->max_helpers = num_helpers;
    hl__kmpc_task->list = (struct helper__kmpc_task_t *)malloc(sizeof(struct helper__kmpc_task_t) * num_helpers);
		if (hl__kmpc_task->list == NULL)
		{
			fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "preallocate_kmpc_helpers: ERROR! Invalid initialization of 'hl__kmpc_task->list' (%d helpers)\n ", THREAD_LEVEL_VAR, num_helpers);
			exit(-1);	
		}
		for (i=0; i<num_helpers; i++)
		{
			hl__kmpc_task->list[i].wrap_task = NULL;
			hl__kmpc_task->list[i].real_task = NULL;
		}
	}

	pthread_mutex_unlock(&hl__kmpc_task_mtx);

	pthread_mutex_lock(&hl__kmpc_taskloop_mtx);

	if (hl__kmpc_taskloop == NULL)
	{
    hl__kmpc_taskloop = (struct helper_list__kmpc_taskloop_t *)malloc(sizeof(struct helper_list__kmpc_taskloop_t));
		if (hl__kmpc_taskloop == NULL)
		{
			fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "preallocate_kmpc_helpers: ERROR! Invalid initialization of 'hl__kmpc_taskloop'\n ", THREAD_LEVEL_VAR);
		  exit(-1);
		}

		hl__kmpc_taskloop->next_id = 0;
		for (i=0; i<MAX_TASKLOOP_HELPERS; i++)
		{
			hl__kmpc_taskloop->real_task_map_by_helper[i] = NULL;
		}
	}

	pthread_mutex_unlock(&hl__kmpc_taskloop_mtx);
}

/**
 * helper__kmpc_task_register
 *
 * Associates a real and a wrapped task in the list of active data helpers.
 *
 * @param wrap_task The wrapper task that substitutes the real one
 * @param real_task The real task that got substituted 
 */
static void helper__kmpc_task_register(void *wrap_task, void *real_task)
{
	int i = 0;

	pthread_mutex_lock(&hl__kmpc_task_mtx);
	if (hl__kmpc_task->count < hl__kmpc_task->max_helpers)
	{
		/* Look for a free slot in the list */
		while (hl__kmpc_task->list[i].wrap_task != NULL) 
		{
			i++;
		}
    /* Add the pair of (wrapper, real) tasks to the assigned slot */
		hl__kmpc_task->list[i].wrap_task = wrap_task;
		hl__kmpc_task->list[i].real_task = real_task;
		hl__kmpc_task->count ++;
#if defined(DEBUG)
		fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_task_register: Registering helper wrap_task=%p real_task=%p slot=%d\n ", THREAD_LEVEL_VAR, wrap_task, real_task, i);
#endif 
	}
	else
	{
    fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_task_register: ERROR! Can not register more tasks because all data helpers are already in use. Please try increasing %s over %d until this error disappears\n ", THREAD_LEVEL_VAR, ENV_VAR_EXTRAE_OPENMP_HELPERS, hl__kmpc_task->max_helpers);
		exit(-1);
	}
	pthread_mutex_unlock(&hl__kmpc_task_mtx);
}

/**
 * helper__kmpc_task_retrieve
 *
 * Retrieves the real_task for the given wrap_task from the list of active data
 * helpers.
 *
 * @param The wrapper task that substitutes the real one
 *
 * @return The real task that got substituted
 */
static void * helper__kmpc_task_retrieve(void *wrap_task)
{
	int i = 0;
	void *real_task = NULL;

	pthread_mutex_lock(&hl__kmpc_task_mtx);

	if (hl__kmpc_task->count > 0)
	{
		while (i < hl__kmpc_task->max_helpers)
		{
	    if (hl__kmpc_task->list[i].wrap_task == wrap_task)
			{
				real_task = hl__kmpc_task->list[i].real_task;

				/* Mark this slot free */
				hl__kmpc_task->list[i].wrap_task = NULL;
				hl__kmpc_task->list[i].real_task = NULL;
			  hl__kmpc_task->count --;
				break;
			}
			i ++;
		}
	}

	pthread_mutex_unlock(&hl__kmpc_task_mtx);

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_task_retrieve: Retrieving helper for wrap_task=%p => real_task=%p\n ", THREAD_LEVEL_VAR, wrap_task, real_task);
#endif 

	if (real_task == NULL)
	{
 		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_task_retrieve: ERROR! Could not free data helper for wrap_task=%p (%d/%d helpers)\n ", THREAD_LEVEL_VAR, wrap_task, hl__kmpc_task->count, hl__kmpc_task->max_helpers);
	}

	return real_task;
}

/**
 * helper__kmpc_task_substitute
 *
 * Callback function that the runtime invokes when a task is going to be
 * executed, where we perform the emission of events and the task substitution
 * from the wrapper to the real one.
 *
 * @param (p1,p2) These are the parameters that the runtime pass to the task to
 * execute, p2 is the task entry pointer to the wrapped task that can be used 
 * to retrieve the real one.
 */
static void helper__kmpc_task_substitute (int arg, void *wrap_task)
{
#if defined(DEBUG)                                                              
  fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_task_substitute enter: args=(%d %p)\n ", THREAD_LEVEL_VAR, arg, wrap_task);
#endif                                                                          

	void (*real_task)(int,void*) = (void(*)(int,void*)) helper__kmpc_task_retrieve (wrap_task);

#if defined(DEBUG)
	fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_task_substitute enter: Substitution for wrap_task=%p is real_task=%p\n ", THREAD_LEVEL_VAR, wrap_task, real_task);
#endif

	if (real_task != NULL)
	{
		Extrae_OpenMP_TaskUF_Entry (real_task);
		real_task (arg, wrap_task); /* Original code execution */
		Extrae_OpenMP_Notify_NewExecutedTask();
		Extrae_OpenMP_TaskUF_Exit ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_task_substitute: ERROR! Did not find task substitution for wrap_task=%p\n ", THREAD_LEVEL_VAR, wrap_task);
		exit (-1);
	}
}

/**
 * helper__kmpc_taskloop_substitute
 *
 * Callback function that the runtime invokes when a task from a taskloop is
 * executed. We retrieve the real task from the helper map indexed by helper_id.
 * The callback is trampolined through the call of taskloop_helper_fn_[0-1023],
 * intermediate functions that interpose as last parameter the helper_id, which
 * identifies the helper function that was invoked. We do this to be able to 
 * retrieve the real task pointer from different tasks from different taskloops
 * that may be executing simultaneously from different threads. 
 *
 * @param arg Argument passed by the runtime.
 * @param wrap_task Argument passed by the runtime. Corresponds to a kmp_task_t 
 * structure.
 * @param helper_id The helper identifier used to retrieve the corresponding
 * real task pointer.
 */
void helper__kmpc_taskloop_substitute (int arg, void *wrap_task, int helper_id)
{
#if defined(DEBUG)                                                              
	  fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_taskloop_substitute enter: args=(%d %p %d)\n ", THREAD_LEVEL_VAR, arg, wrap_task, helper_id);
#endif                                                                          

		void (*real_task)(int,void*) = (void(*)(int,void*)) hl__kmpc_taskloop->real_task_map_by_helper[helper_id];

#if defined(DEBUG)                                                              
	  fprintf(stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_taskloop_substitute enter: Substitution for wrap_task=%p is real_task=%p (helper_id=%d)\n ", THREAD_LEVEL_VAR, wrap_task, real_task, helper_id);
#endif                                                                          

		if (real_task != NULL)
		{
			Extrae_OpenMP_TaskUF_Entry (real_task);
			real_task (arg, wrap_task);
			Extrae_OpenMP_Notify_NewExecutedTask();
			Extrae_OpenMP_TaskUF_Exit ();
		}
		else
		{
			fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "helper__kmpc_taskloop_substitute: ERROR! Did not find task substitution for wrap_task=%p (helper_id=%d)\n ", THREAD_LEVEL_VAR, wrap_task, helper_id);
			exit (-1);
		}
}


/******************************************************************************\
 *                                                                            *
 *                                WRAPPERS                                    *
 *                                                                            *
\******************************************************************************/

void ompc_set_num_threads (int arg)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": ompc_set_num_threads enter: @=%p args=(%d)\n", ompc_set_num_threads_real, arg);
#endif

	RECHECK_INIT(ompc_set_num_threads_real);

	if (TRACE(ompc_set_num_threads_real))
	{
		Backend_ChangeNumberOfThreads (arg);

		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_SetNumThreads_Entry (arg);
		ompc_set_num_threads_real (arg);
		Probe_OpenMP_SetNumThreads_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (ompc_set_num_threads_real != NULL)
	{
		ompc_set_num_threads_real (arg);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": ompc_set_num_threads: ERROR! This function is not hooked! Exiting!!\n");
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": ompc_set_num_threads exit\n");
#endif
}

void __kmpc_barrier (void *loc, int global_tid)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_barrier enter: @=%p args=(%p %d)\n ", THREAD_LEVEL_VAR, __kmpc_barrier_real, loc, global_tid);
#endif

	RECHECK_INIT(__kmpc_barrier_real);

	if (TRACE(__kmpc_barrier_real))
	{
		Extrae_OpenMP_Barrier_Entry ();
		__kmpc_barrier_real (loc, global_tid);
		Extrae_OpenMP_Barrier_Exit ();
	}
	else if (__kmpc_barrier_real != NULL)
	{
		__kmpc_barrier_real (loc, global_tid);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_barrier: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_barrier exit\n ", THREAD_LEVEL_VAR);
#endif
}

void __kmpc_critical (void *loc, int global_tid, void *crit)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_critical enter: @=%p args=(%p %d %p)\n ", THREAD_LEVEL_VAR, __kmpc_critical_real, loc, global_tid, crit);
#endif

	RECHECK_INIT(__kmpc_critical_real);

	if (TRACE(__kmpc_critical_real))
	{
		Extrae_OpenMP_Named_Lock_Entry ();
		__kmpc_critical_real (loc, global_tid, crit);
		Extrae_OpenMP_Named_Lock_Exit (crit);
	}
	else if (__kmpc_critical_real != NULL)
	{
		__kmpc_critical_real (loc, global_tid, crit);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_critical: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_critical exit\n ", THREAD_LEVEL_VAR);
#endif
}

void __kmpc_end_critical (void *loc, int global_tid, void *crit)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_end_critical enter: @=%p args=(%p %d %p)\n ", THREAD_LEVEL_VAR, __kmpc_end_critical_real, loc, global_tid, crit);
#endif

	RECHECK_INIT(__kmpc_end_critical_real);

	if (TRACE(__kmpc_end_critical_real))
	{
		Extrae_OpenMP_Named_Unlock_Entry (crit);
		__kmpc_end_critical_real (loc, global_tid, crit);
		Extrae_OpenMP_Named_Unlock_Exit ();
	}
	else if (__kmpc_end_critical_real != NULL)
	{
		__kmpc_end_critical_real (loc, global_tid, crit);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_end_critical: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_end_critical exit\n ", THREAD_LEVEL_VAR);
#endif
}

void __kmpc_dispatch_init_4 (void *loc, int gtid, int schedule, int lb, int ub, int st, int chunk)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_4 enter: @=%p args=(%p %d %d %d %d %d %d)\n ", THREAD_LEVEL_VAR, __kmpc_dispatch_init_4_real, loc, gtid, schedule, lb, ub, st, chunk);
#endif

	RECHECK_INIT(__kmpc_dispatch_init_4_real);

	if (TRACE(__kmpc_dispatch_init_4_real))
	{
		/* 
		 * Retrieve the outlined function from the parent's thread.
		 * This is executed inside a parallel by multiple threads, so the current worker thread 
		 * retrieves this data from the parent thread who store it at the start of the parallel.
		 */
		struct thread_helper_t *thread_helper = get_parent_thread_helper();
		void *par_uf = thread_helper->par_uf;
#if defined(DEBUG)
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_4: par_uf=%p\n ", THREAD_LEVEL_VAR, par_uf);
#endif

		Extrae_OpenMP_DO_Entry ();

		__kmpc_dispatch_init_4_real (loc, gtid, schedule, lb, ub, st, chunk);
  
		Extrae_OpenMP_UF_Entry (par_uf); 
	}
	else if (__kmpc_dispatch_init_4_real != NULL)
	{
		__kmpc_dispatch_init_4_real (loc, gtid, schedule, lb, ub, st, chunk);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_4: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_4 exit\n ", THREAD_LEVEL_VAR);
#endif
}

void __kmpc_dispatch_init_8 (void *loc, int gtid, int schedule, long long lb, long long ub, long long st, long long chunk)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_8 enter: @=%p args=(%p %d %d %lld %lld %lld %lld)\n ", THREAD_LEVEL_VAR, __kmpc_dispatch_init_8_real, loc, gtid, schedule, lb, ub, st, chunk);
#endif

	RECHECK_INIT(__kmpc_dispatch_init_8_real);

	if (TRACE(__kmpc_dispatch_init_8_real))
	{
		/* 
		 * Retrieve the outlined function from the parent's thread.
		 * This is executed inside a parallel by multiple threads, so the current worker thread 
		 * retrieves this data from the parent thread who store it at the start of the parallel.
		 */
		struct thread_helper_t *thread_helper = get_parent_thread_helper();
		void *par_uf = thread_helper->par_uf;
#if defined(DEBUG)
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_8: par_uf=%p\n ", THREAD_LEVEL_VAR, par_uf);
#endif

		Extrae_OpenMP_DO_Entry ();

		__kmpc_dispatch_init_8_real (loc, gtid, schedule, lb, ub, st, chunk);

		Extrae_OpenMP_UF_Entry (par_uf);
	}
	else if (__kmpc_dispatch_init_8_real != NULL)
	{
		__kmpc_dispatch_init_8_real (loc, gtid, schedule, lb, ub, st, chunk);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_8: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_init_8 exit\n ", THREAD_LEVEL_VAR);
#endif
}

int __kmpc_dispatch_next_4 (void *loc, int gtid, int *p_last, int *p_lb, int *p_ub, int *p_st)
{
	int res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_next_4 enter: @=%p args=(%p %d %p %p %p %p)\n ", THREAD_LEVEL_VAR, __kmpc_dispatch_next_4_real, loc, gtid, p_last, p_lb, p_ub, p_st);
#endif

	RECHECK_INIT(__kmpc_dispatch_next_4_real);

	if (TRACE(__kmpc_dispatch_next_4_real))
	{
		Extrae_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_4_real (loc, gtid, p_last, p_lb, p_ub, p_st);
		Extrae_OpenMP_Work_Exit();

		if (res == 0) /* Alternative to call __kmpc_dispatch_fini_4 which seems not to be called ? */
		{
			Extrae_OpenMP_UF_Exit ();
			Extrae_OpenMP_DO_Exit ();
		}
	}
	else if (__kmpc_dispatch_next_4_real != NULL)
	{
		res = __kmpc_dispatch_next_4_real (loc, gtid, p_last, p_lb, p_ub, p_st);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_next_8: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_next_4 exit: res=%d\n ", THREAD_LEVEL_VAR, res);
#endif

	return res;
}

int __kmpc_dispatch_next_8 (void *loc, int gtid, int *p_last, long long *p_lb, long long *p_ub, long long *p_st)
{
	int res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_next_8 enter: @=%p args=(%p %d %p %p %p %p)\n ", THREAD_LEVEL_VAR, __kmpc_dispatch_next_8_real, loc, gtid, p_last, p_lb, p_ub, p_st);
#endif
	
	RECHECK_INIT(__kmpc_dispatch_next_8_real);

	if (TRACE(__kmpc_dispatch_next_8_real))
	{
		Extrae_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_8_real (loc, gtid, p_last, p_lb, p_ub, p_st);
		Extrae_OpenMP_Work_Exit();

		if (res == 0) /* Alternative to call __kmpc_dispatch_fini_8 which seems not to be called ? */
		{
			Extrae_OpenMP_UF_Exit ();
			Extrae_OpenMP_DO_Exit ();
		}
	}
	else if (__kmpc_dispatch_next_8_real != NULL)
	{
		res = __kmpc_dispatch_next_8_real (loc, gtid, p_last, p_lb, p_ub, p_st);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_next_8: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_next_8 exit: res=%d\n ", THREAD_LEVEL_VAR, res);
#endif

	return res;
}

void __kmpc_dispatch_fini_4 (void *loc, int gtid)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_fini_4 enter: @=%p args=(%p %d)\n ", THREAD_LEVEL_VAR, __kmpc_dispatch_fini_4_real, loc, gtid);
#endif

	RECHECK_INIT(__kmpc_dispatch_fini_4_real);

	if (TRACE(__kmpc_dispatch_fini_4_real))
	{
		Extrae_OpenMP_DO_Exit ();
		__kmpc_dispatch_fini_4_real (loc, gtid);
		Extrae_OpenMP_UF_Exit ();
	}
	else if (__kmpc_dispatch_fini_4_real != NULL)
	{
		__kmpc_dispatch_fini_4_real (loc, gtid);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_fini_4: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_fini_4 exit\n ", THREAD_LEVEL_VAR);
#endif
}

void __kmpc_dispatch_fini_8 (void *loc, int gtid)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_fini_8 enter: @=%p args=(%p %d)\n ", THREAD_LEVEL_VAR, __kmpc_dispatch_fini_8_real, loc, gtid);
#endif

	RECHECK_INIT(__kmpc_dispatch_fini_8_real);

	if (TRACE(__kmpc_dispatch_fini_8_real))
	{
		Extrae_OpenMP_DO_Exit ();
		__kmpc_dispatch_fini_8_real (loc, gtid);
		Extrae_OpenMP_UF_Exit ();
	}
	else if (__kmpc_dispatch_fini_8_real != NULL)
	{
		__kmpc_dispatch_fini_8_real (loc, gtid);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_fini_8: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_dispatch_fini_8 exit\n ", THREAD_LEVEL_VAR);
#endif
}

void __kmpc_fork_call (void *loc, int argc, void *microtask, ...)
{
	void  *args[INTEL_OMP_FUNC_ENTRIES];
	char   kmpc_parallel_wrap_name[1024];
	char   kmpc_parallel_sched_name[1024];
	void (*kmpc_parallel_sched_ptr)(void*,int,void*,void*,void **) = NULL;
	void  *wrap_ptr = NULL;
	void  *task_ptr = microtask;
	va_list ap;
	int     i = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_fork_call enter: @=%p args=(%p %d %p)\n ", THREAD_LEVEL_VAR, __kmpc_fork_call_real, loc, argc, microtask);
#endif

	RECHECK_INIT(__kmpc_fork_call_real);

	if (__kmpc_fork_call_real == NULL)
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_fork_call: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
 		exit (-1);
	}

	/* Grab parameters */
	memset(args, 0, sizeof(args));

	va_start (ap, microtask);
	for (i=0; i<argc; i++)
	{
		args[i] = va_arg(ap, void *);
	}
	va_end (ap);

	/* 
	 * Store the outlined function on this thread's helper. 
	 * This corresponds to the start of a parallel region, 
	 * which is executed by the master thread only.
	 */
	struct thread_helper_t *thread_helper = get_thread_helper();
	thread_helper->par_uf = task_ptr;

	/* Retrieve handler to the scheduling routine that will call __kmpc_fork_call_real with the correct number of arguments */
	snprintf(kmpc_parallel_sched_name, sizeof(kmpc_parallel_sched_name), "__kmpc_parallel_sched_%d_args", argc);
	kmpc_parallel_sched_ptr = (void(*)(void*,int,void*,void*,void **)) dlsym(RTLD_DEFAULT, kmpc_parallel_sched_name);
	if (kmpc_parallel_sched_ptr == NULL)
	{
    fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
										 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
		                 kmpc_parallel_sched_name, argc);
		exit (-1);                                                                  

	}

	if (EXTRAE_ON())
	{
		Extrae_OpenMP_ParRegion_Entry ();

		snprintf(kmpc_parallel_wrap_name, sizeof(kmpc_parallel_wrap_name), "__kmpc_parallel_wrap_%d_args", argc);
		wrap_ptr = dlsym(RTLD_DEFAULT, kmpc_parallel_wrap_name);
		if (wrap_ptr == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		 	                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
											 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
			                 kmpc_parallel_wrap_name, argc);
			exit (-1);                                                                  
		}
	}                      

	/* Call the scheduling routine. 
	 * If wrap_ptr is not NULL, it will interpose a call to wrap_ptr with an extra
	 * parameter with the real task_ptr, in order to instrument when the task
	 * starts executing.
	 */
	kmpc_parallel_sched_ptr(loc, argc, task_ptr, wrap_ptr, args); 

	if (EXTRAE_ON())
	{
		Extrae_OpenMP_ParRegion_Exit ();	
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_fork_call exit\n ", THREAD_LEVEL_VAR);
#endif
}

/**
 * __kmpc_fork_call_dyninst
 *
 *   dlsym() does not seem to work under Dyninst and we can't replace this
 *   function by itself (opposite to MPI, OpenMP does not have something like
 *   PMPI). Thus, we need to pass the address of the original __kmpc_fork_call
 *   (through _extrae_intel_kmpc_init_dyninst) and let the new 
 *   __kmpc_fork_call_dyninst do the work by finally calling to the pointer to
 *   __kmpc_fork_call passed.
 */
void __kmpc_fork_call_dyninst (void *loc, int argc, void *microtask, ...)
{
	void   *args[INTEL_OMP_FUNC_ENTRIES];
	char    kmpc_parallel_wrap_name[1024];
	char    kmpc_parallel_sched_name[1024];
	void  (*kmpc_parallel_sched_ptr)(void*,int,void*,void*,void **) = NULL;
	void   *wrap_ptr = NULL;
	void   *task_ptr = microtask;
	va_list ap;
	int     i = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_fork_call_dyninst enter: @=%p __kmpc_fork_call_real=%p args=(%p %d %p)\n ", THREAD_LEVEL_VAR, __kmpc_fork_call_dyninst, __kmpc_fork_call_real, loc, argc, microtask);
#endif

	RECHECK_INIT(__kmpc_fork_call_real);

	if (__kmpc_fork_call_real == NULL)
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_fork_call_dyninst: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

	/* Grab parameters */
	memset(args, 0, sizeof(args));

	va_start (ap, microtask);
  for (i=0; i<argc; i++)
	{
		args[i] = va_arg(ap, void *);
	}
	va_end (ap);

	/* Retrieve handler to the scheduling routine that will call __kmpc_fork_call_real with the correct number of arguments */

	snprintf(kmpc_parallel_sched_name, sizeof(kmpc_parallel_sched_name), "__kmpc_parallel_sched_%d_args", argc);
  kmpc_parallel_sched_ptr = (void(*)(void*,int,void*,void*,void **)) dlsym(RTLD_DEFAULT, kmpc_parallel_sched_name);
	if (kmpc_parallel_sched_ptr == NULL)                                          
	{
    fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
										 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
		                 kmpc_parallel_sched_name, argc);
		exit (-1);                                                                  
	}

	if (EXTRAE_ON())
	{
		Extrae_OpenMP_ParRegion_Entry ();
		Extrae_OpenMP_EmitTaskStatistics();

		snprintf(kmpc_parallel_wrap_name, sizeof(kmpc_parallel_wrap_name), "__kmpc_parallel_wrap_%d_args", argc);
		wrap_ptr = dlsym(RTLD_DEFAULT, kmpc_parallel_wrap_name);
		if (wrap_ptr == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		 	                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
											 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
			                 kmpc_parallel_wrap_name, argc);
			exit (-1);                                                                  
		}
	}

	/* Call the scheduling routine. 
	 * If wrap_ptr is not NULL, it will interpose a call to wrap_ptr with an extra
	 * parameter with the real task_ptr, in order to instrument when the task
	 * starts executing.
	 */
	kmpc_parallel_sched_ptr(loc, argc, task_ptr, wrap_ptr, args); 

	if (EXTRAE_ON())
	{
		Extrae_OpenMP_ParRegion_Exit ();	
		Extrae_OpenMP_EmitTaskStatistics();
	}
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_fork_call_dyninst exit\n ", THREAD_LEVEL_VAR);
#endif
}

int __kmpc_single (void *loc, int global_tid)
{
	int res = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_single enter: @=%p args=(%p %d)\n ", THREAD_LEVEL_VAR, __kmpc_single_real, loc, global_tid);
#endif

	RECHECK_INIT(__kmpc_single_real);

	if (TRACE(__kmpc_single_real))
	{
		Extrae_OpenMP_Single_Entry ();

		res = __kmpc_single_real (loc, global_tid);

		if (res) /* If the thread entered in the single region, track it */
		{
		/* 
		 * Retrieve the outlined function from the parent's thread.
		 * This is executed inside a parallel by multiple threads, so the current worker thread 
		 * retrieves this data from the parent thread who store it at the start of the parallel.
		 */
  		struct thread_helper_t *thread_helper = get_parent_thread_helper();
	  	void *par_uf = thread_helper->par_uf;
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_single: par_uf=%p\n ", THREAD_LEVEL_VAR, par_uf);
#endif

			Extrae_OpenMP_UF_Entry (par_uf); 
		}
		else
		{
			Extrae_OpenMP_Single_Exit ();
		}
	}
	else if (__kmpc_single_real != NULL)
	{
		res = __kmpc_single_real (loc, global_tid);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_single: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_single exit: res=%d\n ", THREAD_LEVEL_VAR, res);
#endif

	return res;
}

void __kmpc_end_single (void *loc, int global_tid)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_end_single enter: @=%p args=(%p %d)\n ", THREAD_LEVEL_VAR, __kmpc_end_single_real, loc, global_tid);
#endif

	RECHECK_INIT(__kmpc_end_single_real);

	if (TRACE(__kmpc_end_single_real))
	{
		/* This is only executed by the thread that entered the single region */
		Extrae_OpenMP_UF_Exit ();
		__kmpc_end_single_real (loc, global_tid);
		Extrae_OpenMP_Single_Exit ();
	}
	else if (__kmpc_end_single_real != NULL)
	{
		__kmpc_end_single_real (loc, global_tid);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_end_single: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_end_single exit\n ", THREAD_LEVEL_VAR);
#endif
}

void * __kmpc_omp_task_alloc (void *loc, int gtid, int flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds, void *task_entry)
{
	void *res = NULL;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_alloc enter: @=%p args=(%p %d %d %d %d %p)\n ", THREAD_LEVEL_VAR, __kmpc_omp_task_alloc_real, loc, gtid, flags, (int)sizeof_kmp_task_t, (int)sizeof_shareds, task_entry);
#endif

	RECHECK_INIT(__kmpc_omp_task_alloc_real);

	if (TRACE(__kmpc_omp_task_alloc_real))
	{
		Extrae_OpenMP_Task_Entry (task_entry);
		Extrae_OpenMP_Notify_NewInstantiatedTask();
		/* 
		 * We change the task to execute to be the callback helper__kmpc_task_substitute.
		 * The pointer to this new task (wrap_task) is associated to the real task 
		 * with helper__kmpc_task_register. The callback function receives the 
		 * wrap_task pointer by parameter, which will be used to retrieve the
		 * pointer to the real task (see helper__kmpc_task_substitute).
		 */
		res = __kmpc_omp_task_alloc_real (loc, gtid, flags, sizeof_kmp_task_t, sizeof_shareds, helper__kmpc_task_substitute);
		helper__kmpc_task_register (res, task_entry);
		Extrae_OpenMP_Task_Exit ();
	}
	else if (__kmpc_omp_task_alloc_real != NULL)
	{
		res = __kmpc_omp_task_alloc_real (loc, gtid, flags, sizeof_kmp_task_t, sizeof_shareds, task_entry);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_alloc: ERROR! This function is not hooked. Exiting!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}
	
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_alloc exit: res=%p\n ", THREAD_LEVEL_VAR, res);
#endif

	return res;
}

void __kmpc_omp_task_begin_if0 (void *loc, int gtid, void *task)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_begin_if0 enter: @=%p args=(%p %d %p)\n ", THREAD_LEVEL_VAR, __kmpc_omp_task_begin_if0_real, loc, gtid, task);
#endif

	RECHECK_INIT(__kmpc_omp_task_begin_if0_real);

	void (*__kmpc_task_substituted_func)(int,void*) = (void(*)(int,void*)) helper__kmpc_task_retrieve (task);

	if (TRACE(__kmpc_task_substituted_func))
	{
		if (__kmpc_omp_task_begin_if0_real != NULL)
		{
			Extrae_OpenMP_TaskUF_Entry (__kmpc_task_substituted_func);
			Extrae_OpenMP_Notify_NewInstantiatedTask();
			Backend_Leave_Instrumentation();
			__kmpc_omp_task_begin_if0_real (loc, gtid, task);
		}
		else
		{
			fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_begin_if0: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
			exit (-1);
		}
	}
	else if (__kmpc_task_substituted_func != NULL)
	{
		 __kmpc_omp_task_begin_if0_real (loc, gtid, task);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_begin_if0: Did not find task substitution for task=%p\n ", THREAD_LEVEL_VAR, task);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_begin_if0 exit\n ", THREAD_LEVEL_VAR);
#endif
}

void __kmpc_omp_task_complete_if0 (void *loc, int gtid, void *task)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_complete_if0 enter: @=%p args=(%p %d %p)\n ", THREAD_LEVEL_VAR, __kmpc_omp_task_complete_if0_real, loc, gtid, task);
#endif

	RECHECK_INIT(__kmpc_omp_task_complete_if0_real);

	if (TRACE(__kmpc_omp_task_complete_if0_real))
	{
		__kmpc_omp_task_complete_if0_real (loc, gtid, task);
		Extrae_OpenMP_Notify_NewExecutedTask();
		Extrae_OpenMP_TaskUF_Exit ();
	}
	else if (__kmpc_omp_task_complete_if0_real != NULL)
	{
		__kmpc_omp_task_complete_if0_real (loc, gtid, task);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_complete_if0: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_task_complete_if0 exit\n ", THREAD_LEVEL_VAR);
#endif
}

int __kmpc_omp_taskwait (void *loc, int gtid)
{
	int res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_taskwait enter: @=%p args=(%p %d)\n ", THREAD_LEVEL_VAR, __kmpc_omp_taskwait_real, loc, gtid);
#endif

	RECHECK_INIT(__kmpc_omp_taskwait_real);

	if (TRACE(__kmpc_omp_taskwait_real))
	{
		Extrae_OpenMP_Taskwait_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		res = __kmpc_omp_taskwait_real (loc, gtid);
		Extrae_OpenMP_Taskwait_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else if (__kmpc_omp_taskwait_real != NULL)
	{
		res = __kmpc_omp_taskwait_real (loc, gtid);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_taskwait: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_omp_taskwait exit: res=%d\n ", THREAD_LEVEL_VAR, res);
#endif

	return res;
}

/**
 * __kmpc_taskloop
 *
 * When the user application has a taskloop, before the runtime executes this
 * instrumented function, it executes __kmpc_omp_task_alloc, which is also 
 * instrumented. This means that we have captured the allocation of the task,
 * and modified it to invoke the callback helper__kmpc_task_substitute, and
 * added a new entry on the list of (wrapped,real) tasks. The parameter "task"
 * of this function corresponds to the wrapped task, whose routine_entry_ptr
 * field (it's a kmp_task_t struct) points to our callback.
 *
 * However, the runtime internally makes copies of this task, so we can no
 * longer use its pointer to retrieve the real task from our list. To solve
 * this:
 * 1) We retrieve the real task pointer before calling the runtime.
 * 2) We modify the field routine_entry_ptr from the "kmp_task_t task"
 * parameter, directly offsetting the structure, because we have looked in the 
 * libomp how it is implemented. Instead, we assign a pointer to one helper 
 * function, each time one different, so that the tasks executed from different
 * taskloops rely on a separate helper. 
 * 3) We assign helpers incrementally, and the ID of the helper is used to save
 * a map of "helper id" -> "real task pointer".
 * 4) When the runtime executes the task, the helper pointed by
 * routine_entry_ptr will be invoked. These helpers are static routines
 * generated by us, so that each routine passes one last parameter with its own 
 * ID (see intel-kmpc-11-taskloop-helpers.c), and we use the ID to retrieve 
 * the real task pointer.
 *
 * NOTE: Currently, we have support for 1024 helpers, then they start being 
 * reused. So as long as there's no more than 1024 different taskloops going 
 * on simultaneously, this structure should hold.
 */
void __kmpc_taskloop(void *loc, int gtid, void *task, int if_val, void *lb, void *ub, long st, int nogroup, int sched, long grainsize, void *task_dup)
{ 
  int helper_id = 0;	
	void *real_task = NULL;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_taskloop enter: @=%p args=(%p %d %p %d %p %p %ld %d %d %ld %p)\n ", THREAD_LEVEL_VAR, __kmpc_taskloop_real, loc, gtid, task, if_val, lb, ub, st, nogroup, sched, grainsize, task_dup);
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_taskloop: instrumentation is %s\n", THREAD_LEVEL_VAR, (getTrace_OMPTaskloop() ? "enabled" : "disabled"));
#endif

	RECHECK_INIT(__kmpc_taskloop_real);

	/* Retrieve the real task pointer from the list that is maintained in the
	 * instrumented function __kmpc_omp_task_alloc */
	real_task = helper__kmpc_task_retrieve(task);

	if (TRACE(__kmpc_taskloop_real) && (getTrace_OMPTaskloop()))
	{
		Extrae_OpenMP_TaskLoop_Entry ();

		/* Assign a new helper for this taskloop */
		pthread_mutex_lock(&hl__kmpc_taskloop_mtx);
		helper_id = hl__kmpc_taskloop->next_id;
	  hl__kmpc_taskloop->next_id = (hl__kmpc_taskloop->next_id + 1) % MAX_TASKLOOP_HELPERS;
		pthread_mutex_unlock(&hl__kmpc_taskloop_mtx);

		/* Modify the routine_entry_ptr field from the "kmp_task_t task", to point
		 * to the corresponding helper function */
		void **routine_entry_ptr = task + sizeof(void *);
		*routine_entry_ptr = get_taskloop_helper_fn_ptr(helper_id);

		/* Save a map of helper_id => real_task */
		hl__kmpc_taskloop->real_task_map_by_helper[helper_id] = real_task;

		/* Call the runtime */
		__kmpc_taskloop_real(loc, gtid, task, if_val, lb, ub, st, nogroup, sched, grainsize, task_dup);
		Extrae_OpenMP_TaskLoop_Exit ();
	}
	else if (__kmpc_taskloop_real != NULL)
	{
		__kmpc_taskloop_real(loc, gtid, task, if_val, lb, ub, st, nogroup, sched, grainsize, task_dup);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_taskloop: ERROR! This function is not hooked! Exiting!!\n ", THREAD_LEVEL_VAR);
		exit (-1);
	}

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "__kmpc_taskloop exit\n ", THREAD_LEVEL_VAR);
#endif
}

/******************************************************************************\
 *                                                                            *
 *                             INITIALIZATIONS                                *
 *                                                                            *
\******************************************************************************/

/**
 * intel_kmpc_get_hook_points
 *
 * Find the real implementation of the functions. We use dlsym to find the next 
 * definition of the different symbols of the OpenMP runtime (i.e. skip our     
 * wrapper, find the real one). 
 *
 * @param rank The current process ID (not used).
 *
 * @return 1 if any hook was found; 0 otherwise.
 */
static int intel_kmpc_get_hook_points (int rank)
{
	int count = 0;

	UNREFERENCED_PARAMETER(rank);

	/* Obtain @ for ompc_set_num_threads */
	ompc_set_num_threads_real =
		(void(*)(int)) dlsym (RTLD_NEXT, "ompc_set_num_threads");
	INC_IF_NOT_NULL(ompc_set_num_threads_real, count);

	/* Obtain @ for __kmpc_barrier */
	__kmpc_barrier_real =
		(void(*)(void*,int))
		dlsym (RTLD_NEXT, "__kmpc_barrier");
	INC_IF_NOT_NULL(__kmpc_barrier_real,count);

	/* Obtain @ for __kmpc_critical */
	__kmpc_critical_real =
		(void(*)(void*,int,void*))
		dlsym (RTLD_NEXT, "__kmpc_critical");
	INC_IF_NOT_NULL(__kmpc_critical_real,count);

	/* Obtain @ for __kmpc_end_critical */
	__kmpc_end_critical_real =
		(void(*)(void*,int,void*))
		dlsym (RTLD_NEXT, "__kmpc_end_critical");
	INC_IF_NOT_NULL(__kmpc_end_critical_real,count);

	/* Obtain @ for __kmpc_dispatch_init_4 */
	__kmpc_dispatch_init_4_real =
		(void(*)(void*,int,int,int,int,int,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_init_4");
	INC_IF_NOT_NULL(__kmpc_dispatch_init_4_real,count);

	/* Obtain @ for __kmpc_dispatch_init_8 */
	__kmpc_dispatch_init_8_real =
		(void(*)(void*,int,int,long long,long long,long long,long long)) dlsym (RTLD_NEXT, "__kmpc_dispatch_init_8");
	INC_IF_NOT_NULL(__kmpc_dispatch_init_8_real,count);

	/* Obtain @ for __kmpc_dispatch_next_4 */
	__kmpc_dispatch_next_4_real =
		(int(*)(void*,int,int*,int*,int*,int*))
		dlsym (RTLD_NEXT, "__kmpc_dispatch_next_4");
	INC_IF_NOT_NULL(__kmpc_dispatch_next_4_real,count);

	/* Obtain @ for __kmpc_dispatch_next_8 */
	__kmpc_dispatch_next_8_real =
		(int(*)(void*,int,int*,long long *,long long *, long long *))
		dlsym (RTLD_NEXT, "__kmpc_dispatch_next_8");
	INC_IF_NOT_NULL(__kmpc_dispatch_next_8_real,count);

	/* Obtain @ for __kmpc_dispatch_fini_4 */
	__kmpc_dispatch_fini_4_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_fini_4");
	INC_IF_NOT_NULL(__kmpc_dispatch_fini_4_real,count);

	/* Obtain @ for __kmpc_dispatch_fini_8 */
	__kmpc_dispatch_fini_8_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_fini_8");
	INC_IF_NOT_NULL(__kmpc_dispatch_fini_8_real,count);

	/* Obtain @ for __kmpc_fork_call */
	if (__kmpc_fork_call_real == NULL)
	{
		/* Careful, do not overwrite the pointer to the real call if Dyninst has already done it */
		__kmpc_fork_call_real =
			(void(*)(void*,int,void*,...))
			dlsym (RTLD_NEXT, "__kmpc_fork_call");
		INC_IF_NOT_NULL(__kmpc_fork_call_real,count);
	}

	/* Obtain @ for __kmpc_single */
	__kmpc_single_real =
		(int(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_single");
	INC_IF_NOT_NULL(__kmpc_single_real,count);

	/* Obtain @ for __kmpc_end_single */
	__kmpc_end_single_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_end_single");
	INC_IF_NOT_NULL(__kmpc_end_single_real,count);

	/* Obtain @ for __kmpc_omp_task_alloc */
	__kmpc_omp_task_alloc_real =
		(void*(*)(void*,int,int,size_t,size_t,void*)) dlsym (RTLD_NEXT, "__kmpc_omp_task_alloc");
	INC_IF_NOT_NULL(__kmpc_omp_task_alloc_real, count);

	/* Obtain @ for __kmpc_omp_task_begin_if0 */
	__kmpc_omp_task_begin_if0_real =
		(void(*)(void*,int,void*)) dlsym (RTLD_NEXT, "__kmpc_omp_task_begin_if0");
	INC_IF_NOT_NULL(__kmpc_omp_task_begin_if0_real, count);

	/* Obtain @ for __kmpc_omp_task_complete_if0 */
	__kmpc_omp_task_complete_if0_real =
		(void(*)(void*,int,void*)) dlsym (RTLD_NEXT, "__kmpc_omp_task_complete_if0");
	INC_IF_NOT_NULL(__kmpc_omp_task_complete_if0_real, count);

	/* Obtain @ for __kmpc_omp_taskwait */
	__kmpc_omp_taskwait_real = (int(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_omp_taskwait");
	INC_IF_NOT_NULL(__kmpc_omp_taskwait_real, count);

	/* Obtain @ for __kmpc_taskloop */
	__kmpc_taskloop_real = (void(*)(void*,int,void*,int,void*,void*,long,int,int,long,void*)) dlsym (RTLD_NEXT, "__kmpc_taskloop");
	INC_IF_NOT_NULL(__kmpc_taskloop_real, count);

	/* Any hook point? */
	return count > 0;
}

/**
 * _extrae_intel_kmpc_init_dyninst
 *
 * __kmpc_fork_call can not be wrapped as usual with Dyninst because dlsym() 
 * fails to find the real symbol. We pass the pointer to the real function
 * from the Dyninst launcher.
 *
 * @param fork_call_ptr The pointer to the real __kmpc_fork_call.
 */
void _extrae_intel_kmpc_init_dyninst(void *fork_call_ptr)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME ":" THREAD_LEVEL_LBL "_extrae_intel_kmpc_init_dyninst enter: args=(%p)\n ", THREAD_LEVEL_VAR, fork_call_ptr);
#endif

	__kmpc_fork_call_real = (void(*)(void*,int,void*,...)) fork_call_ptr;
}

/**
 * _extrae_intel_kmpc_init
 *
 * Initializes the instrumentation module for Intel KMPC.
 *
 * @param rank The current process ID (not used).
 */
int _extrae_intel_kmpc_init(int rank)
{
	preallocate_kmpc_helpers();

	allocate_nested_helpers();

  return intel_kmpc_get_hook_points(rank);
}

#endif /* PIC */
