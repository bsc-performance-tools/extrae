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

#include "wrapper.h"
#include "trace_macros.h"
#include "omp_probe.h"

//#define DEBUG

static char UNUSED rcsid[] = "$Id$";

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

static void (*__kmpc_fork_call_real)(void*,int,void*,...) = NULL;
static void (*__kmpc_barrier_real)(void*,int) = NULL;
static void (*__kmpc_critical_real)(void*,int,void*) = NULL;
static void (*__kmpc_end_critical_real)(void*,int,void*) = NULL;
static int (*__kmpc_dispatch_next_4_real)(void*,int,int*,int*,int*,int*) = NULL;
static int (*__kmpc_dispatch_next_8_real)(void*,int,int*,long long *,long long *, long long *) = NULL;
static int (*__kmpc_single_real)(void*,int) = NULL;
static void (*__kmpc_end_single_real)(void*,int) = NULL;
#if 0 /* Do not provide information */
static void (*__kmpc_for_static_init_4_real)(void*,int,int,int*,int*,int*,int*,int,int) = NULL;
static void (*__kmpc_for_static_init_8_real)(void*,int,int,int*,long long*,long long*,long long*,long long,long long) = NULL;
static void (*__kmpc_for_static_fini_real)(void*,int) = NULL;
#endif
static void (*__kmpc_dispatch_init_4_real)(void*,int,int,int,int,int,int) = NULL;
static void (*__kmpc_dispatch_init_8_real)(void*,int,int,long long,long long,long long,long long) = NULL;
static void (*__kmpc_dispatch_fini_4_real)(void*,int) = NULL;
static void (*__kmpc_dispatch_fini_8_real)(void*,long long) = NULL; /* Don't sure about this! */

int intel_kmpc_11_hook_points (int rank)
{
	int count = 0;

	/* Obtain @ for __kmpc_fork_call */
	__kmpc_fork_call_real =
		(void(*)(void*,int,void*,...))
		dlsym (RTLD_NEXT, "__kmpc_fork_call");
	if (__kmpc_fork_call_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_fork_call in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_fork_call_real,count);

	/* Obtain @ for __kmpc_barrier */
	__kmpc_barrier_real =
		(void(*)(void*,int))
		dlsym (RTLD_NEXT, "__kmpc_barrier");
	if (__kmpc_barrier_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_barrier in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_barrier_real,count);

	/* Obtain @ for __kmpc_critical */
	__kmpc_critical_real =
		(void(*)(void*,int,void*))
		dlsym (RTLD_NEXT, "__kmpc_critical");
	if (__kmpc_critical_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_critical in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_critical_real,count);

	/* Obtain @ for __kmpc_end_critical */
	__kmpc_end_critical_real =
		(void(*)(void*,int,void*))
		dlsym (RTLD_NEXT, "__kmpc_end_critical");
	if (__kmpc_end_critical_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_end_critical in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_end_critical_real,count);

	/* Obtain @ for __kmpc_dispatch_next_4 */
	__kmpc_dispatch_next_4_real =
		(int(*)(void*,int,int*,int*,int*,int*))
		dlsym (RTLD_NEXT, "__kmpc_dispatch_next_4");
	if (__kmpc_dispatch_next_4_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_dispatch_next_4 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_dispatch_next_4_real,count);

	/* Obtain @ for __kmpc_dispatch_next_8 */
	__kmpc_dispatch_next_8_real =
		(int(*)(void*,int,int*,long long *,long long *, long long *))
		dlsym (RTLD_NEXT, "__kmpc_dispatch_next_8");
	if (__kmpc_dispatch_next_8_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_dispatch_next_8 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_dispatch_next_8_real,count);

	/* Obtain @ for __kmpc_dispatch_next_8 */
	__kmpc_single_real =
		(int(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_single");
	if (__kmpc_single_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_single in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_single_real,count);

	/* Obtain @ for __kmpc_dispatch_next_8 */
	__kmpc_end_single_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_end_single");
	if (__kmpc_end_single_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_end_single in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_end_single_real,count);

#if 0
	/* Obtain @ for __kmpc_for_static_init_4 */
	__kmpc_for_static_init_4_real =
		(void(*)(void*,int,int,int*,int*,int*,int*,int,int)) dlsym (RTLD_NEXT, "__kmpc_for_static_init_4");
	if (__kmpc_for_static_init_4_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_for_static_init_4 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_for_static_init_4_real,count);

	/* Obtain @ for __kmpc_for_static_init_8 */
	__kmpc_for_static_init_8_real =
		(void(*)(void*,int,int,int*,long long*,long long*,long long*,long long,long long)) dlsym (RTLD_NEXT, "__kmpc_for_static_init_8");
	if (__kmpc_for_static_init_8_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_for_static_init_8 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_for_static_init_8_real,count);

	/* Obtain @ for __kmpc_for_static_fini */
	__kmpc_for_static_fini_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_for_static_fini");
	if (__kmpc_for_static_fini_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_for_static_fini in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_for_static_fini_real,count);
#endif

	/* Obtain @ for __kmpc_dispatch_init_4 */
	__kmpc_dispatch_init_4_real =
		(void(*)(void*,int,int,int,int,int,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_init_4");
	if (__kmpc_dispatch_init_4_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_dispatch_init_4 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_dispatch_init_4_real,count);

	/* Obtain @ for __kmpc_dispatch_init_8 */
	__kmpc_dispatch_init_8_real =
		(void(*)(void*,int,int,long long,long long,long long,long long)) dlsym (RTLD_NEXT, "__kmpc_dispatch_init_8");
	if (__kmpc_dispatch_init_8_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_dispatch_init_8 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_dispatch_init_8_real,count);

	/* Obtain @ for __kmpc_dispatch_fini_4 */
	__kmpc_dispatch_fini_4_real =
		(void(*)(void*,int)) dlsym (RTLD_NEXT, "__kmpc_dispatch_fini_4");
	if (__kmpc_dispatch_fini_4_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_dispatch_fini_4 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_dispatch_fini_4_real,count);

	/* Obtain @ for __kmpc_dispatch_fini_8 */
	__kmpc_dispatch_fini_8_real =
		(void(*)(void*,long long)) dlsym (RTLD_NEXT, "__kmpc_dispatch_fini_8");
	if (__kmpc_dispatch_fini_8_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find __kmpc_dispatch_fini_8 in DSOs!!\n");
	INC_IF_NOT_NULL(__kmpc_dispatch_fini_8_real,count);

	/* Any hook point? */
	return count > 0;
}

static void *par_func;

#include "intel-kmpc-11-intermediate.c"

void __kmpc_fork_call (void *p1, int p2, void *p3, ...)
{
	void *params[64];
	va_list ap;
	int i;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_fork_call is at %p\n", THREADID, __kmpc_fork_call_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_fork_call params %p %d %p\n", THREADID, p1, p2, p3);
#endif

	if (__kmpc_fork_call_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_ParRegion_Entry ();

		/* Grab parameters */
		va_start (ap, p3);
		for (i = 0; i < p2; i++)
			params[i] = va_arg (ap, void*);
		va_end (ap);

		par_func = p3;

		switch (p2)
		{
			/* This big switch is handled by this file generated automatically by  genstubs-kmpc-11.sh */
#include "intel-kmpc-11-intermediate-switch.c"

			default:
				fprintf (stderr, PACKAGE_NAME": Error! Unhandled __kmpc_fork_call with %d arguments! Quitting!\n", p2);
				exit (-1);
				break;
		}

		Probe_OpenMP_ParRegion_Exit ();	
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_fork_call is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_barrier (void *p1, int p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_barrier is at %p\n", THREADID, __kmpc_barrier_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_barrier params %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_barrier_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Barrier_Entry ();
		__kmpc_barrier_real (p1, p2);
		Probe_OpenMP_Barrier_Exit ();
		Backend_Leave_Instrumentation ();
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

	if (__kmpc_critical_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Lock_Entry ();
		__kmpc_critical_real (p1, p2, p3);
		Probe_OpenMP_Named_Lock_Exit ();
		Backend_Leave_Instrumentation ();
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

	if (__kmpc_end_critical_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Named_Unlock_Entry ();
		__kmpc_end_critical_real (p1, p2, p3);
		Probe_OpenMP_Named_Unlock_Exit ();
		Backend_Leave_Instrumentation ();
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

	if (__kmpc_dispatch_next_8_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_4_real (p1, p2, p3, p4, p5, p6);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();

		if (res == 0) /* Alternative to call __kmpc_dispatch_fini_4 which seems not to be called ? */
		{
			Backend_Enter_Instrumentation (2);
			Probe_OpenMP_UF_Exit ();
			Probe_OpenMP_DO_Exit ();
			Backend_Leave_Instrumentation ();
		}
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
	if (__kmpc_dispatch_next_8_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_8_real (p1, p2, p3, p4, p5, p6);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();

		if (res == 0) /* Alternative to call __kmpc_dispatch_fini_8 which seems not to be called ? */
		{
			Probe_OpenMP_UF_Exit ();
			Probe_OpenMP_DO_Exit ();
			Backend_Leave_Instrumentation ();
		}
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

	if (__kmpc_single_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_Single_Entry ();
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

	if (__kmpc_single_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		__kmpc_end_single_real (p1, p2);
		Probe_OpenMP_Single_Exit ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": __kmpc_critical is not hooked! exiting!!\n");
		exit (0);
	}
}

#if 0
void  __kmpc_for_static_init_4 (void *p1, int p2, int p3, int *p4, int *p5,
	int *p6, int *p7, int p8, int p9)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_for_static_init_4 is at %p\n", THREADID, __kmpc_for_static_init_4_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_for_static_init_4 params are %p %d %d %p %p %p %p %d %d\n", THREADID, p1, p2, p3, p4, p5, p6, p7, p8, p9);
#endif

	if (__kmpc_for_static_init_4_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
//		Probe_OpenMP_UF_Entry ((UINT64) par_func /*(UINT64)p1*/); /* p1 cannot be translated with bfd? */
		__kmpc_for_static_init_4_real (p1, p2, p3, p4, p5, p6, p7, p8, p9);
		Backend_Enter_Instrumentation (1);
//		Probe_OpenMP_DO_Entry ();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_for_static_init_4 is not hooked! exiting!!\n");
		exit (0);
	}
}

void  __kmpc_for_static_init_8 (void *p1, int p2, int p3, int *p4,
	long long *p5, long long *p6, long long *p7, long long p8, long long p9)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_for_static_init_8 is at %p\n", THREADID, __kmpc_for_static_init_8_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_for_static_init_8 params are %p %d %d %p %p %p %p %lld %lld\n", THREADID, p1, p2, p3, p4, p5, p6, p7, p8, p9);
#endif

	if (__kmpc_for_static_init_8_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
//		Probe_OpenMP_UF_Entry ((UINT64) par_func /*(UINT64)p1*/); /* p1 cannot be translated with bfd? */
		__kmpc_for_static_init_8_real (p1, p2, p3, p4, p5, p6, p7, p8, p9);
		Backend_Enter_Instrumentation (1);
//		Probe_OpenMP_DO_Entry ();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_for_static_init_8 is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_for_static_fini (void *p1, int p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_for_static_fini is at %p\n", THREADID, __kmpc_for_static_fini_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_for_static_fini params are %p %d\n", THREADID, p1, p2);
#endif

	if (__kmpc_for_static_fini_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
//		Probe_OpenMP_DO_Exit ();
		__kmpc_for_static_fini_real (p1, p2);
//		Probe_OpenMP_UF_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_for_static_fini is not hooked! exiting!!\n");
		exit (0);
	}
}
#endif

void __kmpc_dispatch_init_4 (void *p1, int p2, int p3, int p4, int p5, int p6,
	int p7)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_init_4 is at %p\n", THREADID, __kmpc_dispatch_init_4_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_init_4 params are %p %d %d %d %d %d %d\n", THREADID, p1, p2, p3, p4, p5, p6, p7);
#endif

	if (__kmpc_dispatch_init_4_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_DO_Entry ();
		__kmpc_dispatch_init_4_real (p1, p2, p3, p4, p5, p6, p7);
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_UF_Entry ((UINT64) par_func /*(UINT64)p1*/); /* p1 cannot be translated with bfd? */
		Backend_Leave_Instrumentation ();
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

	if (__kmpc_dispatch_init_8_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_DO_Entry ();
		__kmpc_dispatch_init_8_real (p1, p2, p3, p4, p5, p6, p7);
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_UF_Entry ((UINT64) par_func /*(UINT64)p1*/); /* p1 cannot be translated with bfd? */
		Backend_Leave_Instrumentation ();
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

	if (__kmpc_dispatch_fini_4_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_DO_Exit ();
		__kmpc_dispatch_fini_4_real (p1, p2);
		Probe_OpenMP_UF_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_dispatch_fini_4 is not hooked! exiting!!\n");
		exit (0);
	}
}

void __kmpc_dispatch_fini_8 (void *p1, long long p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_fini_8 is at %p\n", THREADID, __kmpc_dispatch_fini_8_real);
	fprintf (stderr, PACKAGE_NAME": THREAD %d: __kmpc_dispatch_fini_8 params are %p %lld\n", THREADID, p1, p2);
#endif

	if (__kmpc_dispatch_fini_8_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_DO_Exit ();
		__kmpc_dispatch_fini_8_real (p1, p2);
		Probe_OpenMP_UF_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME":__kmpc_dispatch_fini_8 is not hooked! exiting!!\n");
		exit (0);
	}
}
