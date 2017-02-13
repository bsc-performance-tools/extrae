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

#include "wrapper.h"
#include "omp_probe.h"
#include "omp-common.h"
#include "intel-kmpc-11-intermediate/intel-kmpc-11-intermediate.h"

//#define DEBUG

#if defined(PIC)

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

struct __kmpv_location_t
{
	int reserved_1;
	int flags;
	int reserved_2;
	int reserved_3;
	char *location;
};

struct __kmp_task_t
{
	void *shareds;
	void *routine;
	int part_id;
};

static pthread_mutex_t extrae_map_kmpc_mutex = PTHREAD_MUTEX_INITIALIZER;

void (*__kmpc_fork_call_real)(void*,int,void*,...) = NULL;
static void (*__kmpc_barrier_real)(void*,int) = NULL;
static void (*__kmpc_critical_real)(void*,int,void*) = NULL;
static void (*__kmpc_end_critical_real)(void*,int,void*) = NULL;
static int (*__kmpc_dispatch_next_4_real)(void*,int,int*,int*,int*,int*) = NULL;
static int (*__kmpc_dispatch_next_8_real)(void*,int,int*,long long *,long long *, long long *) = NULL;
static int (*__kmpc_single_real)(void*,int) = NULL;
static void (*__kmpc_end_single_real)(void*,int) = NULL;
static void (*__kmpc_dispatch_init_4_real)(void*,int,int,int,int,int,int) = NULL;
static void (*__kmpc_dispatch_init_8_real)(void*,int,int,long long,long long,long long,long long) = NULL;
static void (*__kmpc_dispatch_fini_4_real)(void*,int) = NULL;
static void (*__kmpc_dispatch_fini_8_real)(void*,int) = NULL; /* Not sure about this! */

static void* (*__kmpc_omp_task_alloc_real)(void*,int,int,size_t,size_t,void*) = NULL;
static void (*__kmpc_omp_task_begin_if0_real)(void*,int,void*) = NULL;
static void (*__kmpc_omp_task_complete_if0_real)(void*,int,void*) = NULL;
static int (*__kmpc_omp_taskwait_real)(void*,int) = NULL;

static void (*ompc_set_num_threads_real)(int) = NULL;

int intel_kmpc_11_hook_points (int rank)
{
	int count = 0;

	UNREFERENCED_PARAMETER(rank);

	/* Careful, do not overwrite the pointer to the real call if DynInst has
	   already done it */
	if (__kmpc_fork_call_real == NULL)
	{
		/* Obtain @ for __kmpc_fork_call */
		__kmpc_fork_call_real =
			(void(*)(void*,int,void*,...))
			dlsym (RTLD_NEXT, "__kmpc_fork_call");
		INC_IF_NOT_NULL(__kmpc_fork_call_real,count);
	}

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

	/* Obtain @ for __kmpc_dispatch_next_8 */
	__kmpc_single_real =
		(int(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_single");
	INC_IF_NOT_NULL(__kmpc_single_real,count);

	/* Obtain @ for __kmpc_dispatch_next_8 */
	__kmpc_end_single_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_end_single");
	INC_IF_NOT_NULL(__kmpc_end_single_real,count);

	/* Obtain @ for __kmpc_dispatch_init_4 */
	__kmpc_dispatch_init_4_real =
		(void(*)(void*,int,int,int,int,int,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_init_4");
	INC_IF_NOT_NULL(__kmpc_dispatch_init_4_real,count);

	/* Obtain @ for __kmpc_dispatch_init_8 */
	__kmpc_dispatch_init_8_real =
		(void(*)(void*,int,int,long long,long long,long long,long long)) dlsym (RTLD_NEXT, "__kmpc_dispatch_init_8");
	INC_IF_NOT_NULL(__kmpc_dispatch_init_8_real,count);

	/* Obtain @ for __kmpc_dispatch_fini_4 */
	__kmpc_dispatch_fini_4_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_fini_4");
	INC_IF_NOT_NULL(__kmpc_dispatch_fini_4_real,count);

	/* Obtain @ for __kmpc_dispatch_fini_8 */
	__kmpc_dispatch_fini_8_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_fini_8");
	INC_IF_NOT_NULL(__kmpc_dispatch_fini_8_real,count);

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

	/* Obtain @ for ompc_set_num_threads */
	ompc_set_num_threads_real =
		(void(*)(int)) dlsym (RTLD_NEXT, "ompc_set_num_threads");
	INC_IF_NOT_NULL(ompc_set_num_threads_real, count);

	/* Any hook point? */
	return count > 0;
}

/* The par_func variable was used to store the pointer to the task, 
 * but this global variable is not reentrant! So when there are multiple
 * threads, the behavior of the program may change due to threads retrieving
 * an incorrect task.
 * static void *par_func;
 */

void Extrae_intel_kmpc_runtime_init_dyninst (void *fork_call)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME" DEBUG: Extrae_intel_kmpc_runtime_init_dyninst:\n");
	fprintf (stderr, PACKAGE_NAME" DEBUG: fork_call = %p\n", fork_call);
#endif

	__kmpc_fork_call_real = (void(*)(void*,int,void*,...)) fork_call;
}

/*
 * kmpc_fork_call / kmpc_fork_call_extrae_dyninst
 *   dlsym does not seem to work under dyninst and we can't replace this
 *   function by itself (opposite to MPI, OpenMP does not have something like
 *   PMPI). Thus, we need to pass the address of the original __kmpc_fork_call
 *   (through Extrae_intel_kmpc_runtime_init_dyninst) and let the new 
 *   __kmpc_fork_call_extrae_dyninst do the work by finally calling to
 *   __kmpc_fork_call passed.
 */
void __kmpc_fork_call (void *p1, int p2, void *p3, ...)
{
	void  *args[INTEL_OMP_FUNC_ENTRIES];
	char   kmpc_parallel_wrap_name[1024];
	char   kmpc_parallel_sched_name[1024];
	void (*kmpc_parallel_sched_ptr)(void*,int,void*,void*,void **) = NULL;
	void  *wrap_ptr = NULL;
	void  *task_ptr = p3;
	va_list ap;
	int     i = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_fork_call is at %p\n", THREADID, __kmpc_fork_call_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_fork_call params %p %d %p (and more to come ... )\n", THREADID, p1, p2, p3);
#endif

	if (__kmpc_fork_call_real == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_fork_call is not hooked! exiting!!\n");
		exit (-1);
	}

	/* Grab parameters */
	memset(args, 0, sizeof(args));

	va_start (ap, p3);
	for (i=0; i<p2; i++)
	{
		args[i] = va_arg(ap, void *);
	}
	va_end (ap);

	/* Retrieve handler to the scheduling routine that will call __kmpc_fork_call_real with the correct number of arguments */
	snprintf(kmpc_parallel_sched_name, sizeof(kmpc_parallel_sched_name), "__kmpc_parallel_sched_%d_args", p2);
	kmpc_parallel_sched_ptr = (void(*)(void*,int,void*,void*,void **)) dlsym(RTLD_DEFAULT, kmpc_parallel_sched_name);
	if (kmpc_parallel_sched_ptr == NULL)
	{
    fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
										 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
		                 kmpc_parallel_sched_name, p2);
		exit (-1);                                                                  

	}

	if (mpitrace_on)
	{
		Extrae_OpenMP_ParRegion_Entry ();

		snprintf(kmpc_parallel_wrap_name, sizeof(kmpc_parallel_wrap_name), "__kmpc_parallel_wrap_%d_args", p2);
		wrap_ptr = dlsym(RTLD_DEFAULT, kmpc_parallel_wrap_name);
		if (wrap_ptr == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		 	                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
											 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
			                 kmpc_parallel_wrap_name, p2);
			exit (-1);                                                                  
		}
	}                      

	/* Call the scheduling routine. 
	 * If wrap_ptr is not NULL, it will interpose a call to wrap_ptr with an extra
	 * parameter with the real task_ptr, in order to instrument when the task
	 * starts executing.
	 */
	kmpc_parallel_sched_ptr(p1, p2, task_ptr, wrap_ptr, args); 

	if (mpitrace_on)
	{
		Extrae_OpenMP_ParRegion_Exit ();	
	}
}


void __kmpc_fork_call_extrae_dyninst (void *p1, int p2, void *p3, ...)
{
	void   *args[INTEL_OMP_FUNC_ENTRIES];
	char    kmpc_parallel_wrap_name[1024];
	char    kmpc_parallel_sched_name[1024];
	void  (*kmpc_parallel_sched_ptr)(void*,int,void*,void*,void **) = NULL;
	void   *wrap_ptr = NULL;
	void   *task_ptr = p3;
	va_list ap;
	int     i = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_fork_call_extrae_dyninst is at %p\n", THREADID, __kmpc_fork_call_extrae_dyninst);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_fork_call is at %p\n", THREADID, __kmpc_fork_call_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_fork_call params %p %d %p (and more to come ... )\n", THREADID, p1, p2, p3);
#endif

	if (__kmpc_fork_call_real == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_fork_call is not hooked! exiting!!\n");
		exit (0);
	}

	/* Grab parameters */
	memset(args, 0, sizeof(args));

	va_start (ap, p3);
  for (i=0; i<p2; i++)
	{
		args[i] = va_arg(ap, void *);
	}
	va_end (ap);

	/* Retrieve handler to the scheduling routine that will call __kmpc_fork_call_real with the correct number of arguments */

	snprintf(kmpc_parallel_sched_name, sizeof(kmpc_parallel_sched_name), "__kmpc_parallel_sched_%d_args", p2);
  kmpc_parallel_sched_ptr = (void(*)(void*,int,void*,void*,void **)) dlsym(RTLD_DEFAULT, kmpc_parallel_sched_name);
	if (kmpc_parallel_sched_ptr == NULL)                                          
	{
    fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
										 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
		                 kmpc_parallel_sched_name, p2);
		exit (-1);                                                                  
	}

	if (mpitrace_on)
	{
		Extrae_OpenMP_ParRegion_Entry ();
		Extrae_OpenMP_EmitTaskStatistics();

		snprintf(kmpc_parallel_wrap_name, sizeof(kmpc_parallel_wrap_name), "__kmpc_parallel_wrap_%d_args", p2);
		wrap_ptr = dlsym(RTLD_DEFAULT, kmpc_parallel_wrap_name);
		if (wrap_ptr == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Can't retrieve handler to stub '%s' (%d arguments)! Quitting!\n"
		 	                 PACKAGE_NAME":        Recompile Extrae to support this number of arguments!\n"
											 PACKAGE_NAME":        Use src/tracer/wrappers/OMP/genstubs-kmpc-11.sh to do so.\n",
			                 kmpc_parallel_wrap_name, p2);
			exit (-1);                                                                  
		}
	}

	/* Call the scheduling routine. 
	 * If wrap_ptr is not NULL, it will interpose a call to wrap_ptr with an extra
	 * parameter with the real task_ptr, in order to instrument when the task
	 * starts executing.
	 */
	kmpc_parallel_sched_ptr(p1, p2, task_ptr, wrap_ptr, args); 

	if (mpitrace_on)
	{
		Extrae_OpenMP_ParRegion_Exit ();	
		Extrae_OpenMP_EmitTaskStatistics();
	}
}

void __kmpc_barrier (void *p1, int p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_barrier is at %p\n", THREADID, __kmpc_barrier_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_barrier params %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_barrier_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Barrier_Entry ();
		__kmpc_barrier_real (p1, p2);
		Extrae_OpenMP_Barrier_Exit ();
	}
	else if (__kmpc_barrier_real != NULL && mpitrace_on)
	{
		__kmpc_barrier_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_barrier is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_critical (void *p1, int p2, void *p3)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_critical is at %p\n", THREADID, __kmpc_critical_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_critical params %p %d %p\n", THREADID, p1, p2, p3);
#endif

	if (__kmpc_critical_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Named_Lock_Entry ();
		__kmpc_critical_real (p1, p2, p3);
		Extrae_OpenMP_Named_Lock_Exit (p3);
	}
	else if (__kmpc_critical_real != NULL && !mpitrace_on)
	{
		__kmpc_critical_real (p1, p2, p3);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_critical is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_end_critical (void *p1, int p2, void *p3)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_end_critical is at %p\n", THREADID, __kmpc_end_critical_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_end_critical params %p %d %p\n", THREADID, p1, p2, p3);
#endif

	if (__kmpc_end_critical_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Named_Unlock_Entry (p3);
		__kmpc_end_critical_real (p1, p2, p3);
		Extrae_OpenMP_Named_Unlock_Exit ();
	}
	else if (__kmpc_end_critical_real != NULL && !mpitrace_on)
	{
		__kmpc_end_critical_real (p1, p2, p3);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_critical is not hooked! exiting!!\n");
		exit (0);
	}
}

int __kmpc_dispatch_next_4 (void *p1, int p2, int *p3, int *p4, int *p5, int *p6)
{
	int res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_next_4 is at %p\n", THREADID, __kmpc_dispatch_next_4_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_next_4 params %p %d %p %p %p %p\n", THREADID, p1, p2, p3, p4, p5, p6);
#endif

	if (__kmpc_dispatch_next_8_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_4_real (p1, p2, p3, p4, p5, p6);
		Extrae_OpenMP_Work_Exit();

		if (res == 0) /* Alternative to call __kmpc_dispatch_fini_4 which seems not to be called ? */
		{
			Extrae_OpenMP_UF_Exit ();
			Extrae_OpenMP_DO_Exit ();
		}
	}
	else if (__kmpc_dispatch_next_8_real != NULL && !mpitrace_on)
	{
		res = __kmpc_dispatch_next_4_real (p1, p2, p3, p4, p5, p6);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_dispatch_next_8 is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int __kmpc_dispatch_next_8 (void *p1, int p2, int *p3, long long *p4, long long *p5, long long *p6)
{
	int res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_next_8 is at %p\n", THREADID, __kmpc_dispatch_next_8_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_next_8 params %p %d %p %p %p %p\n", THREADID, p1, p2, p3, p4, p5, p6);
#endif
	if (__kmpc_dispatch_next_8_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_8_real (p1, p2, p3, p4, p5, p6);
		Extrae_OpenMP_Work_Exit();

		if (res == 0) /* Alternative to call __kmpc_dispatch_fini_8 which seems not to be called ? */
		{
			Extrae_OpenMP_UF_Exit ();
			Extrae_OpenMP_DO_Exit ();
		}
	}
	else if (__kmpc_dispatch_next_8_real != NULL && !mpitrace_on)
	{
		res = __kmpc_dispatch_next_8_real (p1, p2, p3, p4, p5, p6);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_dispatch_next_8 is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

int __kmpc_single (void *p1, int p2)
{
	int res = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_single is at %p\n", THREADID, __kmpc_single_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_single params %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_single_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Single_Entry ();

		res = __kmpc_single_real (p1, p2);

		if (res) /* If the thread entered in the single region, track it */
		{
			struct __kmpv_location_t *loc = (struct __kmpv_location_t*) p1;
			// printf ("loc->location = %s\n", loc->location);
			Extrae_OpenMP_UF_Entry (loc->location);
		}
		else
		{
			Extrae_OpenMP_Single_Exit ();
		}
	}
	else if (__kmpc_single_real != NULL && !mpitrace_on)
	{
		res = __kmpc_single_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_critical is not hooked! exiting!!\n");
		exit (0);
	}

	return res;
}

void __kmpc_end_single (void *p1, int p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_end_single is at %p\n", THREADID, __kmpc_single_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_end_single params %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_single_real != NULL && mpitrace_on)
	{
		/* This is only executed by the thread that entered the single region */
		Extrae_OpenMP_UF_Exit ();
		__kmpc_end_single_real (p1, p2);
		Extrae_OpenMP_Single_Exit ();
	}
	else if (__kmpc_single_real != NULL && !mpitrace_on)
	{
		__kmpc_end_single_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_critical is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_dispatch_init_4 (void *p1, int p2, int p3, int p4, int p5, int p6,
	int p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_init_4 is at %p\n", THREADID, __kmpc_dispatch_init_4_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_init_4 params are %p %d %d %d %d %d %d\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (__kmpc_dispatch_init_4_real != NULL && mpitrace_on)
	{
		struct __kmpv_location_t *loc = (struct __kmpv_location_t*) p1;

		Extrae_OpenMP_DO_Entry ();
		__kmpc_dispatch_init_4_real (p1, p2, p3, p4, p5, p6, p7);
		/*
		 * Originally the argument here was p1, but this cannot be translated with
		 * bfd. Then it was changed to par_func, but this variable was not reentrant
		 * and could point to another thread's task, so we removed it. Now we use
		 * loc->location, copied from the __kmpc_single wrapper.
		 */
		// printf ("loc->location = %s\n", loc->location);
		Extrae_OpenMP_UF_Entry (loc->location);
	}
	else if (__kmpc_dispatch_init_4_real != NULL && !mpitrace_on)
	{
		__kmpc_dispatch_init_4_real (p1, p2, p3, p4, p5, p6, p7);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_dispatch_init_4 is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_dispatch_init_8 (void *p1, int p2, int p3, long long p4,
	long long p5, long long p6, long long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_init_8 is at %p\n", THREADID, __kmpc_dispatch_init_8_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_init_8 params are %p %d %d %lld %lld %lld %lld\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (__kmpc_dispatch_init_8_real != NULL && mpitrace_on)
	{
		struct __kmpv_location_t *loc = (struct __kmpv_location_t*) p1;

		Extrae_OpenMP_DO_Entry ();
		__kmpc_dispatch_init_8_real (p1, p2, p3, p4, p5, p6, p7);
		/*
		 * Originally the argument here was p1, but this cannot be translated with
		 * bfd. Then it was changed to par_func, but this variable was not reentrant
		 * and could point to another thread's task, so we removed it. Now we use
		 * loc->location, copied from the __kmpc_single wrapper.
		 */
		// printf ("loc->location = %s\n", loc->location);
		Extrae_OpenMP_UF_Entry (loc->location);
	}
	else if (__kmpc_dispatch_init_8_real != NULL && !mpitrace_on)
	{
		__kmpc_dispatch_init_8_real (p1, p2, p3, p4, p5, p6, p7);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_dispatch_init_8 is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_dispatch_fini_4 (void *p1, int p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_fini_4 is at %p\n", THREADID, __kmpc_dispatch_fini_4_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_fini_4 params are %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_dispatch_fini_4_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_DO_Exit ();
		__kmpc_dispatch_fini_4_real (p1, p2);
		Extrae_OpenMP_UF_Exit ();
	}
	else if (__kmpc_dispatch_fini_4_real != NULL && !mpitrace_on)
	{
		__kmpc_dispatch_fini_4_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_dispatch_fini_4 is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_dispatch_fini_8 (void *p1, int p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_fini_8 is at %p\n", THREADID, __kmpc_dispatch_fini_8_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_fini_8 params are %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_dispatch_fini_8_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_DO_Exit ();
		__kmpc_dispatch_fini_8_real (p1, p2);
		Extrae_OpenMP_UF_Exit ();
	}
	else if (__kmpc_dispatch_fini_8_real != NULL && !mpitrace_on)
	{
		__kmpc_dispatch_fini_8_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_dispatch_fini_8 is not hooked! exiting!!\n");
		exit (0);
	}
}

#define EXTRAE_MAP_KMPC_TASK_SIZE 1024*1024

struct extrae_map_kmpc_task_function_st
{
	void *kmpc_task;
	void *function;
};

static struct extrae_map_kmpc_task_function_st extrae_map_kmpc_task_function[EXTRAE_MAP_KMPC_TASK_SIZE];
static unsigned extrae_map_kmpc_count = 0;

static void __extrae_add_kmpc_task_function (void *kmpc_task, void *function)
{
	unsigned u = 0;

	pthread_mutex_lock (&extrae_map_kmpc_mutex);
	if (extrae_map_kmpc_count < EXTRAE_MAP_KMPC_TASK_SIZE)
	{
		/* Look for a free place in the table */
		while (extrae_map_kmpc_task_function[u].kmpc_task != NULL)
			u++;

		/* Add the pair and aggregate to the count */
		extrae_map_kmpc_task_function[u].function = function;
		extrae_map_kmpc_task_function[u].kmpc_task = kmpc_task;
		extrae_map_kmpc_count++;
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": THREAD %d Error number of on-the-fly allocated tasks is above EXTRAE_MAP_KMPC_TASK_SIZE (%s:%d)\n", THREADID,  __FILE__, __LINE__);
		exit (0);
	}
	pthread_mutex_unlock (&extrae_map_kmpc_mutex);
}

static void * __extrae_remove_kmpc_task_function (void *kmpc_task)
{
	void *res = NULL;
	unsigned u = 0;

	pthread_mutex_lock (&extrae_map_kmpc_mutex);
	if (extrae_map_kmpc_count > 0)
	{
		while (u < EXTRAE_MAP_KMPC_TASK_SIZE)
		{
			if (extrae_map_kmpc_task_function[u].kmpc_task == kmpc_task)
			{
				res = extrae_map_kmpc_task_function[u].function;
				extrae_map_kmpc_task_function[u].kmpc_task = NULL;
				extrae_map_kmpc_task_function[u].function = NULL;
				extrae_map_kmpc_count--;
				break;
			}
			u++;
		}
	}
	pthread_mutex_unlock (&extrae_map_kmpc_mutex);

	return res;
}

static void __extrae_kmpc_task_substitute (int p1, void *p2)
{
	void (*__kmpc_task_substituted_func)(int,void*) = (void(*)(int,void*)) __extrae_remove_kmpc_task_function (p2);

	if (__kmpc_task_substituted_func != NULL)
	{
		Extrae_OpenMP_TaskUF_Entry (__kmpc_task_substituted_func);
		Backend_Leave_Instrumentation();
		__kmpc_task_substituted_func (p1, p2); /* Original code execution */
		Extrae_OpenMP_Notify_NewExecutedTask();
		Extrae_OpenMP_TaskUF_Exit ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": THREAD %d did not find task substitution (%s:%d)\n", THREADID,  __FILE__, __LINE__);
		exit (0);
	}
}

void * __kmpc_omp_task_alloc (void *p1, int p2, int p3, size_t p4, size_t p5, void *p6)
{
	void *res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_task_alloc_real is at %p\n", THREADID, __kmpc_omp_task_alloc_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_task_alloc params %p %d %d %d %d %p\n", THREADID, p1, p2, p3, p4, p5, p6);
#endif

	if (__kmpc_omp_task_alloc_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Task_Entry (p6);
		Extrae_OpenMP_Notify_NewInstantiatedTask();
		res = __kmpc_omp_task_alloc_real (p1, p2, p3, p4, p5, __extrae_kmpc_task_substitute);
		__extrae_add_kmpc_task_function (res, p6);
		Extrae_OpenMP_Task_Exit ();
	}
	else if (__kmpc_omp_task_alloc_real != NULL && !mpitrace_on)
	{
		res = __kmpc_omp_task_alloc_real (p1, p2, p3, p4, p5, p6);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": THREAD %d __kmpc_omp_task_alloc was not hooked. Exiting!\n", THREADID);
		exit (0);
	}
	
	return res;
}

void __kmpc_omp_task_begin_if0 (void *p1, int p2, void *p3)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_task_begin_if0_real is at %p\n", THREADID, __kmpc_omp_task_begin_if0_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_task_begin_if0 params %p %d %p\n", THREADID, p1, p2, p3);
#endif

	void (*__kmpc_task_substituted_func)(int,void*) = (void(*)(int,void*)) __extrae_remove_kmpc_task_function (p3);

	if (__kmpc_task_substituted_func != NULL && mpitrace_on)
	{
		if (__kmpc_omp_task_begin_if0_real != NULL)
		{
			Extrae_OpenMP_TaskUF_Entry (__kmpc_task_substituted_func);
			Extrae_OpenMP_Notify_NewInstantiatedTask();
			Backend_Leave_Instrumentation();
			__kmpc_omp_task_begin_if0_real (p1, p2, p3);
		}
		else
		{
			fprintf (stderr, PACKAGE_NAME": __kmpc_omp_task_begin_if0 is not hooked! Exiting!!\n");
			exit (0);
		}
	}
	else if (__kmpc_task_substituted_func != NULL && !mpitrace_on)
	{
		 __kmpc_omp_task_begin_if0_real (p1, p2, p3);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": THREAD %d did not find task substitution (%s:%d)\n", THREADID,  __FILE__, __LINE__);
		exit (0);
	}
}

void __kmpc_omp_task_complete_if0 (void *p1, int p2, void *p3)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_task_complete_if0_real is at %p\n", THREADID, __kmpc_omp_task_complete_if0_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_task_complete_if0 params %p %d %p\n", THREADID, p1, p2, p3);
#endif

	if (__kmpc_omp_task_complete_if0_real != NULL && mpitrace_on)
	{
		__kmpc_omp_task_complete_if0_real (p1, p2, p3);
		Extrae_OpenMP_Notify_NewExecutedTask();
		Extrae_OpenMP_TaskUF_Exit ();
	}
	else if (__kmpc_omp_task_complete_if0_real != NULL && !mpitrace_on)
	{
		__kmpc_omp_task_complete_if0_real (p1, p2, p3);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_omp_task_complete_if0 is not hooked! Exiting!!\n");
		exit (0);
	}
}

int __kmpc_omp_taskwait (void *p1, int p2)
{
	int res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_taskwait_real is at %p\n", THREADID, __kmpc_omp_taskwait_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_omp_taskwait params %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_omp_taskwait_real != NULL)
	{
		Extrae_OpenMP_Taskwait_Entry();
		Extrae_OpenMP_EmitTaskStatistics();
		res = __kmpc_omp_taskwait_real (p1, p2);
		Extrae_OpenMP_Taskwait_Exit();
		Extrae_OpenMP_EmitTaskStatistics();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_omp_taskwait is not hooked! Exiting!!\n");
		exit (0);
	}
	return res;
}

void ompc_set_num_threads (int p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": ompc_set_num_threads is at %p\n", ompc_set_num_threads_real);
	fprintf (stderr, PACKAGE_NAME": ompc_set_num_threads params %d\n", p1);
#endif

	if (ompc_set_num_threads_real != NULL && mpitrace_on)
	{
		Backend_ChangeNumberOfThreads (p1);

		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_SetNumThreads_Entry (p1);
		ompc_set_num_threads_real (p1);
		Probe_OpenMP_SetNumThreads_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (ompc_set_num_threads_real != NULL && !mpitrace_on)
	{
		ompc_set_num_threads_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": ompc_set_num_threads is not hooked! exiting!!\n");
		exit (0);
	}
}

#endif /* PIC */
