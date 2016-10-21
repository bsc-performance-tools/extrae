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

#include "wrapper.h"
#include "omp-common.h"

// #define DEBUG

#if defined(PIC)

struct openmp_task_st
{
	void *p1;
	void *p2;
	void *p3;
	long long task_ctr; /* assigned task counter */
};

/*
The nowait issue:

Some OpenMP clauses may include an optional directive called "nowait".
When specified, a thread won't wait for the others when finishing its 
parallel region. Our callback pointers (pardo_uf, do_uf, par_uf, 
par_single, par_sections) are set by each thread in a parallel region. 
While all threads are simultaneously in the same region, there's no problem.
However, if just one thread moves to the following parallel region and 
modifies this variable, the other threads end up invoking a wrong callback.
So far, I've found this problem in DO clauses, but "nowait" can be specified
in many others. 

Which of the callback pointers below have to be changed for an array of 
function pointers?  All of them, maybe? 
*/

/* Pointer to the user function called by a PARALLEL DO REGION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*pardo_uf)(void*) = NULL;

/* Pointer to the user function called by a PARALLEL REGION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*par_uf)(void*) = NULL;

/* Pointer to the user function called by a PARALLEL SECTION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*parsection_uf)(void*) = NULL;

/*
	callme_parsection (void *p1)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Invokes the required iterations of the
	parallel do loop.
*/
static void callme_parsection (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d callme_parsection: par_section() = %p p1 = %p\n", THREADID, parsection_uf, p1);
#endif

	if (parsection_uf == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Invalid initialization of 'par_section'\n");
		exit (0);
	}

	Extrae_OpenMP_UF_Entry (parsection_uf);
	Backend_setInInstrumentation (THREADID, FALSE); /* We're about to execute user code */
	parsection_uf (p1);
	Backend_setInInstrumentation (THREADID, TRUE); /* We're about to execute OpenMP code */
	Extrae_OpenMP_UF_Exit ();
}
/*
	callme_pardo (void *p1)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Invokes the required iterations of the
	parallel do loop.
*/
static void callme_pardo (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d callme_pardo: pardo_uf() = %p p1 = %p\n", THREADID, pardo_uf, p1);
#endif

	if (pardo_uf == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Invalid initialization of 'pardo_uf'\n");
		exit (0);
	}

	Extrae_OpenMP_UF_Entry (pardo_uf);
	Backend_setInInstrumentation (THREADID, FALSE); /* We're about to execute user code */
	pardo_uf (p1);
	Backend_setInInstrumentation (THREADID, TRUE); /* We're about to execute OpenMP code */
	Extrae_OpenMP_UF_Exit ();
}

/*
	callme_par (void *)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Each thread runs the very same routine
	with different params.
*/
static void callme_par (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d callme_par: par_uf()=%p p1=%p\n", THREADID, par_uf, p1);
#endif

	if (par_uf == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Invalid initialization of 'par_uf'\n");
		exit (0);
	}

	Extrae_OpenMP_UF_Entry (par_uf);
	Backend_setInInstrumentation (THREADID, FALSE); /* We're about to execute user code */
	par_uf (p1);
	Backend_setInInstrumentation (THREADID, TRUE); /* We're back to execute OpenMP code */
	Extrae_OpenMP_UF_Exit ();
}

static void (*GOMP_parallel_real)(void*,void*,unsigned,unsigned) = NULL;
static void (*GOMP_parallel_start_real)(void*,void*,unsigned) = NULL;
static void (*GOMP_parallel_end_real)(void) = NULL;
static void (*GOMP_barrier_real)(void) = NULL;
static void (*GOMP_critical_name_start_real)(void**) = NULL;
static void (*GOMP_critical_name_end_real)(void**) = NULL;
static void (*GOMP_critical_start_real)(void) = NULL;
static void (*GOMP_critical_end_real)(void) = NULL;
static void (*GOMP_atomic_start_real)(void) = NULL;
static void (*GOMP_atomic_end_real)(void) = NULL;
static void (*GOMP_parallel_loop_static_start_real)(void*,void*,unsigned, long, long, long, long) = NULL;
static void (*GOMP_parallel_loop_runtime_start_real)(void*,void*,unsigned, long, long, long, long) = NULL;
static void (*GOMP_parallel_loop_dynamic_start_real)(void*,void*,unsigned, long, long, long, long) = NULL;
static void (*GOMP_parallel_loop_guided_start_real)(void*,void*,unsigned, long, long, long, long) = NULL;
static int (*GOMP_loop_static_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_runtime_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_dynamic_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_guided_next_real)(long*,long*) = NULL;
static int (*GOMP_loop_static_start_real)(long,long,long,long,long*,long*) = NULL;
static int (*GOMP_loop_runtime_start_real)(long,long,long,long,long*,long*) = NULL;
static int (*GOMP_loop_guided_start_real)(long,long,long,long,long*,long*) = NULL;
static int (*GOMP_loop_dynamic_start_real)(long,long,long,long,long*,long*) = NULL;
static void (*GOMP_loop_end_real)(void) = NULL;
static void (*GOMP_loop_end_nowait_real)(void) = NULL;
static unsigned (*GOMP_sections_start_real)(unsigned) = NULL;
static unsigned (*GOMP_sections_next_real)(void) = NULL;
static void (*GOMP_sections_end_real)(void) = NULL;
static void (*GOMP_sections_end_nowait_real)(void) = NULL;
static void (*GOMP_parallel_sections_start_real)(void*,void*,unsigned,unsigned) = NULL;
static void (*GOMP_task_real)(void*,void*,void*,long,long,int,unsigned) = NULL;
static void (*GOMP_taskwait_real)(void) = NULL;
static int (*GOMP_loop_ordered_static_start_real)(long, long, long, long, long *, long *) = NULL;
static int (*GOMP_loop_ordered_runtime_start_real)(long, long, long, long, long *, long *) = NULL;
static int (*GOMP_loop_ordered_dynamic_start_real)(long, long, long, long, long *, long *) = NULL;
static int (*GOMP_loop_ordered_guided_start_real)(long, long, long, long, long *, long *) = NULL;
#if 0
/* These seem unnecessary */
static void (*GOMP_ordered_start_real)(void) = NULL;
static void (*GOMP_ordered_end_real)(void) = NULL;
#endif

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

static int gnu_libgomp_4_2_GetOpenMPHookPoints (int rank)
{
	int count = 0;

	UNREFERENCED_PARAMETER(rank);

	/* Obtain @ for GOMP_parallel */
	GOMP_parallel_real =
		(void(*)(void*,void*,unsigned,unsigned)) dlsym (RTLD_NEXT, "GOMP_parallel");
	INC_IF_NOT_NULL(GOMP_parallel_start_real,count);

	/* Obtain @ for GOMP_parallel_start */
	GOMP_parallel_start_real =
		(void(*)(void*,void*,unsigned)) dlsym (RTLD_NEXT, "GOMP_parallel_start");
	INC_IF_NOT_NULL(GOMP_parallel_start_real,count);

	/* Obtain @ for GOMP_parallel_end */
	GOMP_parallel_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_parallel_end");
	INC_IF_NOT_NULL(GOMP_parallel_end_real,count);

	/* Obtain @ for GOMP_barrier */
	GOMP_barrier_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_barrier");
	INC_IF_NOT_NULL(GOMP_barrier_real,count);

	/* Obtain @ for GOMP_atomic_start */
	GOMP_atomic_start_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_atomic_start");
	INC_IF_NOT_NULL(GOMP_atomic_start_real,count);

	/* Obtain @ for GOMP_atomic_end */
	GOMP_atomic_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_atomic_end");
	INC_IF_NOT_NULL(GOMP_atomic_end_real,count);

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

	/* Obtain @ for GOMP_parallel_loop_static_start */
	GOMP_parallel_loop_static_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_static_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_static_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_runtime_start */
	GOMP_parallel_loop_runtime_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_runtime_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_runtime_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_guided_start */
	GOMP_parallel_loop_guided_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_guided_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_guided_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_dynamic_start */
	GOMP_parallel_loop_dynamic_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_dynamic_start");
	INC_IF_NOT_NULL(GOMP_parallel_loop_dynamic_start_real,count);

	/* Obtain @ for GOMP_loop_static_next */
	GOMP_loop_static_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_static_next");
	INC_IF_NOT_NULL(GOMP_loop_static_next_real,count);

	/* Obtain @ for GOMP_loop_runtime_next */
	GOMP_loop_runtime_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_runtime_next");
	INC_IF_NOT_NULL(GOMP_loop_runtime_next_real,count);

	/* Obtain @ for GOMP_loop_guided_next */
	GOMP_loop_guided_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_guided_next");
	INC_IF_NOT_NULL(GOMP_loop_guided_next_real,count);

	/* Obtain @ for GOMP_loop_dynamic_next */
	GOMP_loop_dynamic_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_dynamic_next");
	INC_IF_NOT_NULL(GOMP_loop_dynamic_next_real,count);

	/* Obtain @ for GOMP_loop_static_start */
	GOMP_loop_static_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_static_start");
	INC_IF_NOT_NULL(GOMP_loop_static_start_real,count);

	/* Obtain @ for GOMP_loop_runtime_start */
	GOMP_loop_runtime_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_runtime_start");
	INC_IF_NOT_NULL(GOMP_loop_runtime_start_real,count);

	/* Obtain @ for GOMP_loop_guided_start */
	GOMP_loop_guided_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_guided_start");
	INC_IF_NOT_NULL(GOMP_loop_guided_start_real,count);

	/* Obtain @ for GOMP_loop_dynamic_start */
	GOMP_loop_dynamic_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_dynamic_start");
	INC_IF_NOT_NULL(GOMP_loop_dynamic_start_real,count);

	/* Obtain @ for GOMP_loop_end */
	GOMP_loop_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_loop_end");
	INC_IF_NOT_NULL(GOMP_loop_end_real,count);

	/* Obtain @ for GOMP_loop_end_nowait */
	GOMP_loop_end_nowait_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_loop_end_nowait");
	INC_IF_NOT_NULL(GOMP_loop_end_nowait_real,count);

	/* Obtain @ for GOMP_sections_end */
	GOMP_sections_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_end");
	INC_IF_NOT_NULL(GOMP_sections_end_real,count);

	/* Obtain @ for GOMP_sections_end_nowait */
	GOMP_sections_end_nowait_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_end_nowait");
	INC_IF_NOT_NULL(GOMP_sections_end_nowait_real,count);

	/* Obtain @ for GOMP_sections_start */
	GOMP_sections_start_real =
		(unsigned(*)(unsigned)) dlsym (RTLD_NEXT, "GOMP_sections_start");
	INC_IF_NOT_NULL(GOMP_sections_start_real,count);

	/* Obtain @ for GOMP_sections_next */
	GOMP_sections_next_real =
		(unsigned(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_next");
	INC_IF_NOT_NULL(GOMP_sections_next_real,count);

	/* Obtain @ for GOMP_parallel_sections_start */
	GOMP_parallel_sections_start_real = 
		(void(*)(void*,void*,unsigned,unsigned)) dlsym (RTLD_NEXT, "GOMP_parallel_sections_start");
	INC_IF_NOT_NULL(GOMP_parallel_sections_start_real,count);

	/* Obtain @ for GOMP_task */
	GOMP_task_real =
		(void(*)(void*,void*,void*,long,long,int,unsigned)) dlsym (RTLD_NEXT, "GOMP_task");
	INC_IF_NOT_NULL(GOMP_task_real,count);

	/* Obtain @ for GOMP_taskwait */
	GOMP_taskwait_real = (void(*)(void)) dlsym (RTLD_NEXT, "GOMP_taskwait");
	INC_IF_NOT_NULL(GOMP_taskwait_real,count);

	/* Obtain @ for GOMP_loop_ordered_static_start */
	GOMP_loop_ordered_static_start_real = 
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_static_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_static_start_real, count);

	/* Obtain @ for GOMP_loop_ordered_runtime_start */
	GOMP_loop_ordered_runtime_start_real = 
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_runtime_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_runtime_start_real, count);

	/* Obtain @ for GOMP_loop_ordered_dynamic_start */
	GOMP_loop_ordered_dynamic_start_real = 
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_dynamic_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_dynamic_start_real, count);

	/* Obtain @ for GOMP_loop_ordered_guided_start */
	GOMP_loop_ordered_guided_start_real = 
		(int(*)(long, long, long, long, long *, long *)) dlsym (RTLD_NEXT, "GOMP_loop_ordered_guided_start");
	INC_IF_NOT_NULL(GOMP_loop_ordered_guided_start_real, count);

#if 0
	/* Obtain @ for GOMP_ordered_start */
	GOMP_ordered_start_real = (void(*)(void)) dlsym (RTLD_NEXT, "GOMP_ordered_start");
	INC_IF_NOT_NULL(GOMP_ordered_start_real,count);
	
	/* Obtain @ for GOMP_ordered_end */
	GOMP_ordered_end_real = (void(*)(void)) dlsym (RTLD_NEXT, "GOMP_ordered_end");
	INC_IF_NOT_NULL(GOMP_ordered_end_real,count);
#endif
		
	/* Any hook point? */
	return count > 0;
}

/*
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
*/

static void callme_task (void *helper_ptr)
{
	struct openmp_task_st *helper = (* (struct openmp_task_st**)helper_ptr);
	void (*task_uf)(void*) = (void(*)(void*)) helper->p1;

	if (helper != NULL)
	{
		Extrae_OpenMP_TaskUF_Entry (helper->p1);
		Extrae_OpenMP_TaskID (helper->task_ctr);

		task_uf (helper->p2);
		if (helper->p3 != NULL)
			free(helper->p3);
		free(helper);

		Extrae_OpenMP_Notify_NewExecutedTask();
		Extrae_OpenMP_TaskUF_Exit ();
	}
}

static volatile long long __GOMP_task_ctr = 1;
#if !defined(HAVE__SYNC_FETCH_AND_ADD)
static pthread_mutex_t __GOMP_task_ctr_mtx = PTHREAD_MUTEX_INITIALIZER;
#endif
void GOMP_task (void *p1, void *p2, void *p3, long p4, long p5, int p6, unsigned p7)
{
	void (*task_cpy)(void*,void*) = (void(*)(void*,void*)) p3;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_task is at %p\n", THREADID, GOMP_task_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_task params %p %p %p %ld %ld %ld %d %u\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_task_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Task_Entry (p1);
		Extrae_OpenMP_Notify_NewInstantiatedTask();

		struct openmp_task_st *helper = NULL;
		helper = (struct openmp_task_st *) malloc(sizeof(struct openmp_task_st));
		helper->p1 = p1;
		helper->p2 = p2;

		if (p3 != NULL)
		{
			char *buf =  malloc(sizeof(char) * (p4 + p5 - 1));
			char *arg = (char *) (((uintptr_t) buf + p5 - 1)
			            & ~(uintptr_t) (p5 - 1));
			task_cpy (arg, helper->p2);
			helper->p2 = arg;
			// saved for deallocation purposes, arg is not valid since includes offset
			helper->p3 = buf; 
		}
		else
		{
			char * buf =  malloc(sizeof(char) * (p4 + p5 - 1));
			memcpy (buf, p2, p4);
			helper->p2 = buf;
			// saved for deallocation purposes, arg is not valid since includes offset
			helper->p3 = buf;
		}

#if defined(HAVE__SYNC_FETCH_AND_ADD)
		helper->task_ctr = __sync_fetch_and_add(&__GOMP_task_ctr,1);
#else
		pthread_mutex_lock (&__GOMP_task_ctr_mtx);
		helper->task_ctr = __GOMP_task_ctr++;
		pthread_mutex_unlock (&__GOMP_task_ctr_mtx);
#endif
		Extrae_OpenMP_TaskID (helper->task_ctr);
		GOMP_task_real (callme_task, &helper, NULL, sizeof(helper), p5, p6, p7);
		Extrae_OpenMP_Task_Exit ();
	}
	else if (GOMP_task_real != NULL)
	{
		GOMP_task_real (p1, p2, p3, p4, p5, p6, p7);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_task is not hooked! Exiting!!\n");
		exit (0);
	}
}

void GOMP_taskwait (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_taskwait is at %p\n", THREADID, GOMP_taskwait_real);
#endif

	if (GOMP_taskwait_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Taskwait_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		GOMP_taskwait_real ();
		Extrae_OpenMP_Taskwait_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else if (GOMP_taskwait_real != NULL)
	{
		GOMP_taskwait_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_taskwait is not hooked! Exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_sections_start (void *p1, void *p2, unsigned p3, unsigned p4)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_sections_start is at %p\n", THREADID, GOMP_sections_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_sections params %p %p %u %u \n", THREADID, p1, p2, p3, p4);
#endif

	if (GOMP_parallel_sections_start_real != NULL && EXTRAE_INITIALIZED())
	{
		parsection_uf = (void(*)(void*))p1;

		Extrae_OpenMP_ParSections_Entry();
		GOMP_parallel_sections_start_real (callme_parsection, p2, p3, p4);

		/* The master thread continues the execution and then calls pardo_uf */
		if (THREADID == 0)
			Extrae_OpenMP_UF_Entry (p1);

		/* Extrae_OpenMP_ParSections_Exit(); */
	}
	else if (GOMP_parallel_sections_start_real != NULL)
	{
		GOMP_parallel_sections_start_real (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel_sections_start is not hooked! exiting!!\n");
		exit (0);
	}
}

unsigned GOMP_sections_start (unsigned p1)
{
	unsigned res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_sections_start is at %p\n", THREADID, GOMP_sections_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_sections params %u\n", THREADID, p1);
#endif

	if (GOMP_sections_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Section_Entry();
		res = GOMP_sections_start_real (p1);
		Extrae_OpenMP_Section_Exit();
	}
	else if (GOMP_sections_start_real != NULL)
	{
		res = GOMP_sections_start_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_sections_start is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

unsigned GOMP_sections_next (void)
{
	unsigned res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_sections_next is at %p\n", THREADID, GOMP_sections_next_real);
#endif

	if (GOMP_sections_next_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Work_Entry();
		res = GOMP_sections_next_real();
		Extrae_OpenMP_Work_Exit();
	}
	else if (GOMP_sections_next_real != NULL)
	{
		res = GOMP_sections_next_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_sections_next is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

void GOMP_sections_end (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_sections_end is at %p\n", THREADID, GOMP_sections_end_real);
#endif

	if (GOMP_sections_end_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Join_Wait_Entry();
		GOMP_sections_end_real();
		Extrae_OpenMP_Join_Wait_Exit();
	}
	else if (GOMP_sections_end_real != NULL)
	{
		GOMP_sections_end_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_sections_end is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_sections_end_nowait (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_sections_end_nowait is at %p\n", THREADID, GOMP_sections_end_nowait_real);
#endif

	if (GOMP_sections_end_nowait_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Join_NoWait_Entry();
		GOMP_sections_end_nowait_real();
		Extrae_OpenMP_Join_NoWait_Exit();
	}
	else if (GOMP_sections_end_nowait_real != NULL)
	{
		GOMP_sections_end_nowait_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_sections_end_nowait is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_loop_end (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_end is at %p\n", THREADID, GOMP_loop_end_real);
#endif

	if (GOMP_loop_end_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Join_Wait_Entry();
		GOMP_loop_end_real();
		Extrae_OpenMP_Join_Wait_Exit();
		Extrae_OpenMP_UF_Exit ();
		Extrae_OpenMP_DO_Exit ();	
	}
	else if (GOMP_loop_end_real != NULL)
	{
		GOMP_loop_end_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_end is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_loop_end_nowait (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_end_nowait is at %p\n", THREADID, GOMP_loop_end_nowait_real);
#endif

	if (GOMP_loop_end_nowait_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Join_NoWait_Entry();
		GOMP_loop_end_nowait_real();
		Extrae_OpenMP_Join_NoWait_Exit();
		Extrae_OpenMP_UF_Exit ();
		Extrae_OpenMP_DO_Exit ();	
	}
	else if (GOMP_loop_end_nowait_real != NULL)
	{
		GOMP_loop_end_nowait_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_end_nowait is not hooked! exiting!!\n");
		exit (0);
	}
}

int GOMP_loop_static_start (long p1, long p2, long p3, long p4, long *p5, long *p6)
{
	int res = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_static_start is at %p\n", THREADID, GOMP_loop_static_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %ld %ld %ld %ld %p %p\n", THREADID, p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_static_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_static_start_real (p1, p2, p3, p4, p5, p6);
		Extrae_OpenMP_UF_Entry (par_uf);
		Backend_Leave_Instrumentation();
	}
	else if (GOMP_loop_static_start_real != NULL)
	{
		res = GOMP_loop_static_start_real (p1, p2, p3, p4, p5, p6);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_static_start is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_runtime_start (long p1, long p2, long p3, long p4, long *p5, long *p6)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_runtime_start is at %p\n", THREADID, GOMP_loop_runtime_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %ld %ld %ld %ld %p %p\n", THREADID, p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_runtime_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_runtime_start_real (p1, p2, p3, p4, p5, p6);
		Extrae_OpenMP_UF_Entry (par_uf);
		Backend_Leave_Instrumentation();
	}
	else if (GOMP_loop_runtime_start_real != NULL)
	{
		res = GOMP_loop_runtime_start_real (p1, p2, p3, p4, p5, p6);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_runtime_start is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_guided_start (long p1, long p2, long p3, long p4, long *p5, long *p6)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_guided_start is at %p\n", THREADID, GOMP_loop_guided_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %ld %ld %ld %ld %p %p\n", THREADID, p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_guided_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_guided_start_real (p1, p2, p3, p4, p5, p6);
		Extrae_OpenMP_UF_Entry (par_uf);
		Backend_Leave_Instrumentation();
	}
	else if (GOMP_loop_guided_start_real != NULL)
	{
		res = GOMP_loop_guided_start_real (p1, p2, p3, p4, p5, p6);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_guided_start is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_dynamic_start (long p1, long p2, long p3, long p4, long *p5, long *p6)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_dynamic_start is at %p\n", THREADID, GOMP_loop_dynamic_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %ld %ld %ld %ld %p %p\n", THREADID, p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_dynamic_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_dynamic_start_real (p1, p2, p3, p4, p5, p6);
		Extrae_OpenMP_UF_Entry (par_uf);
		Backend_Leave_Instrumentation();
	}
	else if (GOMP_loop_dynamic_start_real != NULL)
	{
		res = GOMP_loop_dynamic_start_real (p1, p2, p3, p4, p5, p6);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_dynamic_start is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

void GOMP_parallel_loop_static_start (void *p1, void *p2, unsigned p3, long p4, long p5, long p6, long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_loop_static_start is at %p\n", THREADID, GOMP_parallel_loop_static_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p %u %ld %ld %ld %ld\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_static_start_real != NULL && EXTRAE_INITIALIZED())
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Extrae_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_static_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Extrae_OpenMP_ParDO_Exit ();	

		/* The master thread continues the execution and then calls pardo_uf */
		if (THREADID == 0)
			Extrae_OpenMP_UF_Entry (pardo_uf);
	}
	else if (GOMP_parallel_loop_static_start_real != NULL)
	{
		GOMP_parallel_loop_static_start_real (p1, p2, p3, p4, p5, p6, p7);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_static_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_loop_runtime_start (void *p1, void *p2, unsigned p3, long p4, long p5, long p6, long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_loop_runtime_start is at %p\n", THREADID, GOMP_parallel_loop_runtime_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p %u %ld %ld %ld %ld\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_runtime_start_real != NULL && EXTRAE_INITIALIZED())
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Extrae_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_runtime_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Extrae_OpenMP_ParDO_Exit ();	

		/* The master thread continues the execution and then calls pardo_uf */
		if (THREADID == 0)
			Extrae_OpenMP_UF_Entry (pardo_uf);
	}
	else if (GOMP_parallel_loop_runtime_start_real != NULL)
	{
		GOMP_parallel_loop_runtime_start_real (p1, p2, p3, p4, p5, p6, p7);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_runtime_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_loop_guided_start (void *p1, void *p2, unsigned p3, long p4, long p5, long p6, long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_loop_guided_start is at %p\n", THREADID, GOMP_parallel_loop_guided_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p %u %ld %ld %ld %ld\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_static_start_real != NULL && EXTRAE_INITIALIZED())
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Extrae_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_guided_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Extrae_OpenMP_ParDO_Exit ();	

		/* The master thread continues the execution and then calls pardo_uf */
		if (THREADID == 0)
			Extrae_OpenMP_UF_Entry (pardo_uf);
	}
	else if (GOMP_parallel_loop_static_start_real != NULL)
	{
		GOMP_parallel_loop_guided_start_real (p1, p2, p3, p4, p5, p6, p7);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_guided_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_loop_dynamic_start (void *p1, void *p2, unsigned p3, long p4, long p5, long p6, long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_loop_dynamic_start is at %p\n", THREADID, GOMP_parallel_loop_dynamic_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p %u %ld %ld %ld %ld\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_dynamic_start_real != NULL && EXTRAE_INITIALIZED())
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Extrae_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_dynamic_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Extrae_OpenMP_ParDO_Exit ();	

		/* The master thread continues the execution and then calls pardo_uf */
		if (THREADID == 0)
			Extrae_OpenMP_UF_Entry (pardo_uf);
	}
	else if (GOMP_parallel_loop_dynamic_start_real != NULL)
	{
		GOMP_parallel_loop_dynamic_start_real (p1, p2, p3, p4, p5, p6, p7);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_dynamic_start is not hooked! exiting!!\n");
		exit (0);
	}
}

int GOMP_loop_static_next (long *p1, long *p2)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_static_next is at %p\n", THREADID, GOMP_loop_static_next_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p\n", THREADID, p1, p2);
#endif

	if (GOMP_loop_static_next_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Work_Entry();
		res = GOMP_loop_static_next_real (p1, p2);
		Extrae_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_static_next_real != NULL)
	{
		res = GOMP_loop_static_next_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_static_next is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_runtime_next (long *p1, long *p2)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_runtime_next is at %p\n", THREADID, GOMP_loop_runtime_next_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p\n", THREADID, p1, p2);
#endif

	if (GOMP_loop_runtime_next_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Work_Entry();
		res = GOMP_loop_runtime_next_real (p1, p2);
		Extrae_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_runtime_next_real != NULL)
	{
		res = GOMP_loop_runtime_next_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_runtime_next is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_guided_next (long *p1, long *p2)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_guided_next is at %p\n", THREADID, GOMP_loop_guided_next_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p\n", THREADID, p1, p2);
#endif

	if (GOMP_loop_guided_next_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Work_Entry();
		res = GOMP_loop_guided_next_real (p1, p2);
		Extrae_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_guided_next_real != NULL)
	{
		res = GOMP_loop_guided_next_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_guided_next is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_dynamic_next (long *p1, long *p2)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_dynamic_next is at %p\n", THREADID, GOMP_loop_dynamic_next_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d params %p %p\n", THREADID, p1, p2);
#endif

	if (GOMP_loop_dynamic_next_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Work_Entry();
		res = GOMP_loop_dynamic_next_real (p1, p2);
		Extrae_OpenMP_Work_Exit();
	}
	else if (GOMP_loop_dynamic_next_real != NULL)
	{
		res = GOMP_loop_dynamic_next_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_dynamic_next is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

void GOMP_parallel (void *p1, void *p2, unsigned p3, unsigned p4)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel is at %p\n", THREADID, GOMP_parallel_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel params %p %p %u %u\n", THREADID, p1, p2, p3, p4);
#endif

	if (GOMP_parallel_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_ParRegion_Entry();
		Extrae_OpenMP_EmitTaskStatistics();

		/* Set the pointer to the correct PARALLEL user function */
		par_uf = (void(*)(void*))p1;
		GOMP_parallel_real (callme_par, p2, p3, p4);

		Extrae_OpenMP_ParRegion_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else if (GOMP_parallel_real != NULL)
	{
		GOMP_parallel_real (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_start (void *p1, void *p2, unsigned p3)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_start is at %p\n", THREADID, GOMP_parallel_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_start params %p %p %u\n", THREADID, p1, p2, p3);
#endif

	if (GOMP_parallel_start_real != NULL && EXTRAE_INITIALIZED())
	{
		/* Set the pointer to the correct PARALLEL user function */
		par_uf = (void(*)(void*))p1;

		Extrae_OpenMP_ParRegion_Entry();
		Extrae_OpenMP_EmitTaskStatistics();

		GOMP_parallel_start_real (callme_par, p2, p3);

		/* GCC/libgomp does not execute callme_par per root thread, emit
		   the required event here - call Backend to get a new time! */
		Extrae_OpenMP_UF_Entry (p1);
	}
	else if (GOMP_parallel_start_real != NULL)
	{
		GOMP_parallel_start_real (p1, p2, p3);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_end (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_parallel_end is at %p\n", THREADID, GOMP_parallel_end_real);
#endif

	if (GOMP_parallel_end_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_UF_Exit ();
		GOMP_parallel_end_real ();
		Extrae_OpenMP_ParRegion_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else if (GOMP_parallel_end_real != NULL)
	{
		GOMP_parallel_end_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_parallel_end is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_barrier (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_barrier is at %p\n", THREADID, GOMP_barrier_real);
#endif

	if (GOMP_barrier_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Barrier_Entry ();
		GOMP_barrier_real ();
		Extrae_OpenMP_Barrier_Exit ();
	}
	else if (GOMP_barrier_real != NULL)
	{
		GOMP_barrier_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_barrier is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_critical_name_start (void **p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_critical_name_start is at %p\n", THREADID, GOMP_critical_name_start_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_critical_name_start params %p\n", THREADID, p1);
#endif

	if (GOMP_critical_name_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Named_Lock_Entry();
		GOMP_critical_name_start_real (p1);
		Extrae_OpenMP_Named_Lock_Exit(p1);
	}
	else if (GOMP_critical_name_start_real != NULL)
	{
		GOMP_critical_name_start_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_critical_name_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_critical_name_end (void **p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_critical_name_end is at %p\n", THREADID, GOMP_critical_name_end_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_critical_name_end params %p\n", THREADID, p1);
#endif

	if (GOMP_critical_name_end_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Named_Unlock_Entry(p1);
		GOMP_critical_name_end_real (p1);
		Extrae_OpenMP_Named_Unlock_Exit();
	}
	else if (GOMP_critical_name_end_real != NULL)
	{
		GOMP_critical_name_end_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_critical_name_end is not hooked! exiting!!\n");
		exit (0);
	}
}


void GOMP_critical_start (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_critical_start is at %p\n", THREADID, GOMP_critical_start_real);
#endif

	if (GOMP_critical_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Unnamed_Lock_Entry();
		GOMP_critical_start_real();
		Extrae_OpenMP_Unnamed_Lock_Exit();
	}
	else if (GOMP_critical_start_real != NULL)
	{
		GOMP_critical_start_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_critical_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_critical_end (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_critical_end is at %p\n", THREADID, GOMP_critical_end_real);
#endif

	if (GOMP_critical_end_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Unnamed_Unlock_Entry();
		GOMP_critical_end_real ();
		Extrae_OpenMP_Unnamed_Unlock_Exit();
	}
	else if (GOMP_critical_end_real != NULL)
	{
		GOMP_critical_end_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_critical_end is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_atomic_start (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_atomic_start is at %p\n", THREADID, GOMP_atomic_start_real);
#endif

	if (GOMP_atomic_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Unnamed_Lock_Entry();
		GOMP_atomic_start_real();
		Extrae_OpenMP_Unnamed_Lock_Exit();
	}
	else if (GOMP_atomic_start_real != NULL)
	{
		GOMP_atomic_start_real();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_atomic_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_atomic_end (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_atomic_end is at %p\n", THREADID, GOMP_atomic_end_real);
#endif

	if (GOMP_atomic_end_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_Unnamed_Unlock_Entry();
		GOMP_atomic_end_real ();
		Extrae_OpenMP_Unnamed_Unlock_Exit();
	}
	else if (GOMP_atomic_end_real != NULL)
	{
		GOMP_atomic_end_real ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_atomic_end is not hooked! exiting!!\n");
		exit (0);
	}
}

int GOMP_loop_ordered_static_start (long start, long end, long incr,
	long chunk_size, long *istart, long *iend)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_ordered_static_start %p\n", THREADID, GOMP_loop_ordered_static_start_real);
#endif

	if (GOMP_loop_ordered_static_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_static_start_real (start, end, incr, chunk_size, istart, iend);
		Extrae_OpenMP_UF_Entry (par_uf);
	}
	else if (GOMP_loop_ordered_static_start_real != NULL)
	{
		res = GOMP_loop_ordered_static_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_ordered_static_start_real is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_ordered_runtime_start (long start, long end, long incr,
	long chunk_size, long *istart, long *iend)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_ordered_runtime_start %p\n", THREADID, GOMP_loop_ordered_runtime_start_real);
#endif

	if (GOMP_loop_ordered_runtime_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_runtime_start_real (start, end, incr, chunk_size, istart, iend);
		Extrae_OpenMP_UF_Entry (par_uf);
	}
	else if (GOMP_loop_ordered_runtime_start_real != NULL)
	{
		res = GOMP_loop_ordered_runtime_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_ordered_runtime_start_real is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_ordered_dynamic_start (long start, long end, long incr,
	long chunk_size, long *istart, long *iend)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_ordered_dynamic_start %p\n", THREADID, GOMP_loop_ordered_dynamic_start_real);
#endif

	if (GOMP_loop_ordered_dynamic_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_dynamic_start_real (start, end, incr, chunk_size, istart, iend);
		Extrae_OpenMP_UF_Entry (par_uf);
	}
	else if (GOMP_loop_ordered_dynamic_start_real != NULL)
	{
		res = GOMP_loop_ordered_dynamic_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_ordered_dynamic_start_real is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int GOMP_loop_ordered_guided_start (long start, long end, long incr,
	long chunk_size, long *istart, long *iend)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_loop_ordered_guided_start %p\n", THREADID, GOMP_loop_ordered_guided_start_real);
#endif

	if (GOMP_loop_ordered_guided_start_real != NULL && EXTRAE_INITIALIZED())
	{
		Extrae_OpenMP_DO_Entry ();
		res = GOMP_loop_ordered_guided_start_real (start, end, incr, chunk_size, istart, iend);
		Extrae_OpenMP_UF_Entry (par_uf);
	}
	else if (GOMP_loop_ordered_guided_start_real != NULL)
	{
		res = GOMP_loop_ordered_guided_start_real (start, end, incr, chunk_size, istart, iend);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_ordered_guided_start_real is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

#if 0
/* These seem unnecessary */
void GOMP_ordered_start (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_ordered_start is at %p\n", THREADID, GOMP_ordered_start_real);
#endif
	GOMP_ordered_start_real ();
}

void GOMP_ordered_end (void)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d GOMP_ordered_end is at %p\n", THREADID, GOMP_ordered_end_real);
#endif
	GOMP_ordered_end_real();
}
#endif

extern int omp_get_max_threads();

int gnu_libgomp_4_2_hook_points (int ntask)
{
	return gnu_libgomp_4_2_GetOpenMPHookPoints (ntask);
}

#endif /* PIC */
