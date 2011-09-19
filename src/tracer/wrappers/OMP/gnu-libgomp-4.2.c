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
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#if !defined(DYNINST_MODULE)

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

#include "wrapper.h"
#include "trace_macros.h"
#include "omp_probe.h"

/* #define DEBUG */

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

/* FIXME: Should we discover this dinamically? */ 
#define MAX_THD 32


/* Pointer to the user function called by a PARALLEL DO REGION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*pardo_uf)(void*);

/* Pointer to the user function called by a PARALLEL REGION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*par_uf)(void*);


/*
	callme_pardo (void *p1)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Invokes the required iterations of the
	parallel do loop.
*/
static void callme_pardo (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": callme_pardo: p1 = %p\n", p1);
#endif

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) par_uf);
	pardo_uf (p1);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

/*
	callme_par (void *)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Each thread runs the very same routine
	with different params.
*/
static void callme_par (void *ptr)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": callme_par: ptr=%p\n", ptr);
	fprintf (stderr, PACKAGE_NAME": callme_par: par_uf=%p\n", par_uf);
#endif

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) par_uf);
	par_uf (ptr);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

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

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

static int gnu_libgomp_4_2_GetOpenMPHookPoints (int rank)
{
	int count = 0;

	/* Obtain @ for GOMP_parallel_start */
	GOMP_parallel_start_real =
		(void(*)(void*,void*,unsigned)) dlsym (RTLD_NEXT, "GOMP_parallel_start");
	if (GOMP_parallel_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_parallel_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_parallel_start_real,count);

	/* Obtain @ for GOMP_parallel_end */
	GOMP_parallel_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_parallel_end");
	if (GOMP_parallel_end_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_parallel_end in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_parallel_end_real,count);

	/* Obtain @ for GOMP_barrier */
	GOMP_barrier_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_barrier");
	if (GOMP_barrier_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_barrier in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_barrier_real,count);

	/* Obtain @ for GOMP_atomic_start */
	GOMP_atomic_start_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_atomic_start");
	if (GOMP_atomic_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_atomic_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_atomic_start_real,count);

	/* Obtain @ for GOMP_atomic_end */
	GOMP_atomic_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_atomic_end");
	if (GOMP_atomic_end_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_atomic_end in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_atomic_end_real,count);

	/* Obtain @ for GOMP_critical_enter */
	GOMP_critical_start_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_critical_start");
	if (GOMP_critical_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_critical_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_critical_start_real,count);

	/* Obtain @ for GOMP_critical_end */
	GOMP_critical_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_critical_end");
	if (GOMP_critical_end_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_critical_end in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_critical_end_real,count);

	/* Obtain @ for GOMP_critical_name_start */
	GOMP_critical_name_start_real =
		(void(*)(void**)) dlsym (RTLD_NEXT, "GOMP_critical_name_start");
	if (GOMP_critical_name_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_critical_name_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_critical_name_start_real,count);

	/* Obtain @ for GOMP_critical_name_end */
	GOMP_critical_name_end_real =
		(void(*)(void**)) dlsym (RTLD_NEXT, "GOMP_critical_name_end");
	if (GOMP_critical_name_end_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_critical_name_end in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_critical_name_end_real,count);

	/* Obtain @ for GOMP_parallel_loop_static_start */
	GOMP_parallel_loop_static_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_static_start");
	if (GOMP_parallel_loop_static_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_parallel_loop_static_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_parallel_loop_static_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_runtime_start */
	GOMP_parallel_loop_runtime_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_runtime_start");
	if (GOMP_parallel_loop_runtime_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_parallel_loop_runtime_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_parallel_loop_runtime_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_guided_start */
	GOMP_parallel_loop_guided_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_guided_start");
	if (GOMP_parallel_loop_guided_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_parallel_loop_guided_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_parallel_loop_guided_start_real,count);

	/* Obtain @ for GOMP_parallel_loop_dynamic_start */
	GOMP_parallel_loop_dynamic_start_real =
		(void(*)(void*,void*,unsigned, long, long, long, long)) dlsym (RTLD_NEXT, "GOMP_parallel_loop_dynamic_start");
	if (GOMP_parallel_loop_dynamic_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_parallel_loop_dynamic_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_parallel_loop_dynamic_start_real,count);

	/* Obtain @ for GOMP_loop_static_next */
	GOMP_loop_static_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_static_next");
	if (GOMP_loop_static_next_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_static_next in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_static_next_real,count);

	/* Obtain @ for GOMP_loop_runtime_next */
	GOMP_loop_runtime_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_runtime_next");
	if (GOMP_loop_runtime_next_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_runtime_next in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_runtime_next_real,count);

	/* Obtain @ for GOMP_loop_guided_next */
	GOMP_loop_guided_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_guided_next");
	if (GOMP_loop_guided_next_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_guided_next in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_guided_next_real,count);

	/* Obtain @ for GOMP_loop_dynamic_next */
	GOMP_loop_dynamic_next_real =
		(int(*)(long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_dynamic_next");
	if (GOMP_loop_dynamic_next_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_dynamic_next in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_dynamic_next_real,count);

	/* Obtain @ for GOMP_loop_static_start */
	GOMP_loop_static_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_static_start");
	if (GOMP_loop_static_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_static_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_static_start_real,count);

	/* Obtain @ for GOMP_loop_runtime_start */
	GOMP_loop_runtime_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_runtime_start");
	if (GOMP_loop_runtime_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_runtime_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_runtime_start_real,count);

	/* Obtain @ for GOMP_loop_guided_start */
	GOMP_loop_guided_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_guided_start");
	if (GOMP_loop_guided_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_guided_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_guided_start_real,count);

	/* Obtain @ for GOMP_loop_dynamic_start */
	GOMP_loop_dynamic_start_real =
		(int(*)(long,long,long,long,long*,long*)) dlsym (RTLD_NEXT, "GOMP_loop_dynamic_start");
	if (GOMP_loop_dynamic_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_dynamic_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_dynamic_start_real,count);

	/* Obtain @ for GOMP_loop_end */
	GOMP_loop_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_loop_end");
	if (GOMP_loop_end_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_end in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_end_real,count);

	/* Obtain @ for GOMP_loop_end_nowait */
	GOMP_loop_end_nowait_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_loop_end_nowait");
	if (GOMP_loop_end_nowait_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_loop_end_nowait in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_loop_end_nowait_real,count);

	/* Obtain @ for GOMP_sections_end */
	GOMP_sections_end_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_end");
	if (GOMP_sections_end_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_sections_end in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_sections_end_real,count);

	/* Obtain @ for GOMP_sections_end_nowait */
	GOMP_sections_end_nowait_real =
		(void(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_end_nowait");
	if (GOMP_sections_end_nowait_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_sections_end_nowait in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_sections_end_nowait_real,count);

	/* Obtain @ for GOMP_sections_start */
	GOMP_sections_start_real =
		(unsigned(*)(unsigned)) dlsym (RTLD_NEXT, "GOMP_sections_start");
	if (GOMP_sections_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_sections_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_sections_start_real,count);

	/* Obtain @ for GOMP_sections_next */
	GOMP_sections_next_real =
		(unsigned(*)(void)) dlsym (RTLD_NEXT, "GOMP_sections_next");
	if (GOMP_sections_next_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_sections_next in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_sections_next_real,count);

	/* Obtain @ for GOMP_parallel_sections_start */
	GOMP_parallel_sections_start_real = 
		(void(*)(void*,void*,unsigned,unsigned)) dlsym (RTLD_NEXT, "GOMP_parallel_sections_start");
	if (GOMP_parallel_sections_start_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find GOMP_parallel_sections_start in DSOs!!\n");
	INC_IF_NOT_NULL(GOMP_parallel_sections_start_real,count);

	/* Any hook point? */
	return count > 0;
}

/*
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
*/

void GOMP_parallel_sections_start (void *p1, void *p2, unsigned p3, unsigned p4)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_sections_start is at %p\n", GOMP_sections_start_real);
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_sections params %p %p %u %u \n", p1, p2, p3, p4);
#endif

	if (GOMP_parallel_sections_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_ParSections_Entry();
		GOMP_parallel_sections_start_real (p1, p2, p3, p4);
		Probe_OpenMP_ParSections_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_sections_start is at %p\n", GOMP_sections_start_real);
	fprintf (stderr, PACKAGE_NAME": GOMP_sections params %u\n", p1);
#endif

	if (GOMP_sections_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Section_Entry();
		res = GOMP_sections_start_real (p1);
		Probe_OpenMP_Section_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_sections_next is at %p\n", GOMP_sections_next_real);
#endif

	if (GOMP_sections_next_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Work_Entry();
		res = GOMP_sections_next_real();
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_sections_end is at %p\n", GOMP_sections_end_real);
#endif

	if (GOMP_sections_end_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Join_Wait_Entry();
		GOMP_sections_end_real();
		Probe_OpenMP_Join_Wait_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_sections_end_nowait is at %p\n", GOMP_sections_end_nowait_real);
#endif

	if (GOMP_sections_end_nowait_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Join_NoWait_Entry();
		GOMP_sections_end_nowait_real();
 		Probe_OpenMP_Join_NoWait_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_end is at %p\n", GOMP_loop_end_real);
#endif

	if (GOMP_loop_end_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Join_Wait_Entry();
		GOMP_loop_end_real();
		Probe_OpenMP_Join_Wait_Exit();
		Probe_OpenMP_DO_Exit ();	
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_end_nowait is at %p\n", GOMP_loop_end_nowait_real);
#endif

	if (GOMP_loop_end_nowait_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Join_NoWait_Entry();
		GOMP_loop_end_nowait_real();
		Probe_OpenMP_Join_NoWait_Exit();
		Probe_OpenMP_DO_Exit ();	
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_static_start is at %p\n", GOMP_loop_static_start_real);
	fprintf (stderr, PACKAGE_NAME": params %ld %ld %ld %ld %p %p\n", p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_static_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_static_start_real (p1, p2, p3, p4, p5, p6);
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_runtime_start is at %p\n", GOMP_loop_runtime_start_real);
	fprintf (stderr, PACKAGE_NAME": params %ld %ld %ld %ld %p %p\n", p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_runtime_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_runtime_start_real (p1, p2, p3, p4, p5, p6);
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_static_start is at %p\n", GOMP_loop_guided_start_real);
	fprintf (stderr, PACKAGE_NAME": params %ld %ld %ld %ld %p %p\n", p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_guided_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_guided_start_real (p1, p2, p3, p4, p5, p6);
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_dynamic_start is at %p\n", GOMP_loop_dynamic_start_real);
	fprintf (stderr, PACKAGE_NAME": params %ld %ld %ld %ld %p %p\n", p1, p2, p3, p4, p5, p6);
#endif

	if (GOMP_loop_dynamic_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_DO_Entry ();
		res = GOMP_loop_dynamic_start_real (p1, p2, p3, p4, p5, p6);
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_static_start is at %p\n", GOMP_parallel_loop_static_start_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p %u %ld %ld %ld %ld\n", p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_static_start_real != NULL)
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_static_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Probe_OpenMP_ParDO_Exit ();	
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_static_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_loop_runtime_start (void *p1, void *p2, unsigned p3, long p4, long p5, long p6, long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_runtime_start is at %p\n", GOMP_parallel_loop_runtime_start_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p %u %ld %ld %ld %ld\n", p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_runtime_start_real != NULL)
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_runtime_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Probe_OpenMP_ParDO_Exit ();	
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_runtime_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_loop_guided_start (void *p1, void *p2, unsigned p3, long p4, long p5, long p6, long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_guided_start is at %p\n", GOMP_parallel_loop_guided_start_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p %u %ld %ld %ld %ld\n", p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_static_start_real != NULL)
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_guided_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Probe_OpenMP_ParDO_Exit ();	
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_guided_start is not hooked! exiting!!\n");
		exit (0);
	}
}

void GOMP_parallel_loop_dynamic_start (void *p1, void *p2, unsigned p3, long p4, long p5, long p6, long p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_loop_dynamic_start is at %p\n", GOMP_parallel_loop_dynamic_start_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p %u %ld %ld %ld %ld\n", p1, p2, p3, p4, p5, p6, p7);
#endif

	if (GOMP_parallel_loop_dynamic_start_real != NULL)
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(void*))p1;

		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_ParDO_Entry ();
		GOMP_parallel_loop_dynamic_start_real (callme_pardo, p2, p3, p4, p5, p6, p7);
		Probe_OpenMP_ParDO_Exit ();	
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_dynamic_start is not hooked! exiting!!\n");
		exit (0);
	}
}

int GOMP_loop_static_next (long *p1, long *p2)
{
	int res = 0;
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_static_next is at %p\n", GOMP_loop_static_next_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p\n", p1, p2);
#endif

	if (GOMP_loop_static_next_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_static_next_real (p1, p2);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_runtime_next is at %p\n", GOMP_loop_runtime_next_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p\n", p1, p2);
#endif

	if (GOMP_loop_runtime_next_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_runtime_next_real (p1, p2);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_guided_next is at %p\n", GOMP_loop_guided_next_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p\n", p1, p2);
#endif

	if (GOMP_loop_guided_next_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_guided_next_real (p1, p2);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_loop_dynamic_next is at %p\n", GOMP_loop_dynamic_next_real);
	fprintf (stderr, PACKAGE_NAME": params %p %p\n", p1, p2);
#endif

	if (GOMP_loop_dynamic_next_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Work_Entry();
		res = GOMP_loop_dynamic_next_real (p1, p2);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_loop_dynamic_next is not hooked! exiting!!\n");
		exit (0);
	}
	return res;
}

extern int omp_get_thread_num();

void GOMP_parallel_start (void *p1, void *p2, unsigned p3)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_start is at %p\n", GOMP_parallel_start_real);
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_start params %p %p %u\n", p1, p2, p3);
#endif

	if (GOMP_parallel_start_real != NULL)
	{
		/* Set the pointer to the correct PARALLEL user function */
		par_uf = (void(*)(void*))p1;

		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_ParRegion_Entry();

		/* GCC/libgomp does not execute callme_par per root thread, emit
		   the required event here */
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_UF_Entry ((UINT64) p1);

		GOMP_parallel_start_real (callme_par, p2, p3);
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_parallel_end is at %p\n", GOMP_parallel_end_real);
#endif

	if (GOMP_parallel_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_UF_Exit ();
		GOMP_parallel_end_real ();
		Probe_OpenMP_ParRegion_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_barrier is at %p\n", GOMP_barrier_real);
#endif

	if (GOMP_barrier_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Barrier_Entry ();
		GOMP_barrier_real ();
		Probe_OpenMP_Barrier_Exit ();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_critical_name_start is at %p\n", GOMP_critical_name_start_real);
	fprintf (stderr, PACKAGE_NAME": GOMP_critical_name_start params %p\n", p1);
#endif

	if (GOMP_critical_name_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Named_Lock_Entry();
		GOMP_critical_name_start_real (p1);
		Probe_OpenMP_Named_Lock_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_critical_name_end is at %p\n", GOMP_critical_name_end_real);
	fprintf (stderr, PACKAGE_NAME": GOMP_critical_name_end params %p\n", p1);
#endif

	if (GOMP_critical_name_end_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Named_Unlock_Entry();
		GOMP_critical_name_end_real (p1);
		Probe_OpenMP_Named_Unlock_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_critical_start is at %p\n", GOMP_critical_start_real);
#endif

	if (GOMP_critical_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Unnamed_Lock_Entry();
		GOMP_critical_start_real();
		Probe_OpenMP_Unnamed_Lock_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_critical_end is at %p\n", GOMP_critical_end_real);
#endif

	if (GOMP_critical_end_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Unnamed_Unlock_Entry();
		GOMP_critical_end_real ();
		Probe_OpenMP_Unnamed_Unlock_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_atomic_start is at %p\n", GOMP_atomic_start_real);
#endif

	if (GOMP_atomic_start_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Unnamed_Lock_Entry();
		GOMP_atomic_start_real();
		Probe_OpenMP_Unnamed_Lock_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": GOMP_atomic_end is at %p\n", GOMP_atomic_end_real);
#endif

	if (GOMP_atomic_end_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Unnamed_Unlock_Entry();
		GOMP_atomic_end_real ();
		Probe_OpenMP_Unnamed_Unlock_Exit();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": GOMP_atomic_end is not hooked! exiting!!\n");
		exit (0);
	}
}

extern int omp_get_max_threads();

int gnu_libgomp_4_2_hook_points (int ntask)
{
	int hooked;
	int max_threads;

	hooked = gnu_libgomp_4_2_GetOpenMPHookPoints (ntask);
   
	max_threads = omp_get_max_threads();
	if (max_threads > MAX_THD)
	{
		/* Has this happened? */
		/* a) Increase MAX_THD to be higher than omp_get_max_threads() */
		/* b) Decrease OMP_NUM_THREADS in order to decrease omp_get_max_threads() */
		fprintf (stderr, PACKAGE_NAME": omp_get_max_threads() > MAX_THD. Aborting...\nRecompile "PACKAGE_NAME" increasing MAX_THD or decrease OMP_NUM_THREADS\n");
		exit (1);
	}

	return hooked;
}

#endif /* !defined(DYNINST_MODULE) */
