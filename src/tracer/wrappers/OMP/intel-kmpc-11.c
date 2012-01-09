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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/tracer/wrappers/OMP/ibm-xlsmp-1.6.c $
 | @last_commit: $Date: 2011-06-17 10:11:42 +0200 (dv, 17 jun 2011) $
 | @version:     $Revision: 659 $
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

static char UNUSED rcsid[] = "$Id: ibm-xlsmp-1.6.c 659 2011-06-17 08:11:42Z harald $";

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

static void (*__kmpc_fork_call_real)(void*,int,void*,...) = NULL;
static void (*__kmpc_barrier_real)(void*,int) = NULL;
static void (*__kmpc_critical_real)(void*,int,void*) = NULL;
static void (*__kmpc_end_critical_real)(void*,int,void*) = NULL;
static int (*__kmpc_dispatch_next_4_real)(void*,int,int*,int*,int*,int*) = NULL;
static int (*__kmpc_dispatch_next_8_real)(void*,int,int*,long long *,long long *, long long *) = NULL;
static int (*__kmpc_single_real)(void*,int) = NULL;
static void (*__kmpc_end_single_real)(void*,int) = NULL;

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

	/* Any hook point? */
	return count > 0;
}

static void *par_func;

static void __kmpc_parallel_func_8param (int *p1, int *p2, void *param1, void *param2, void *param3, void *param4, void *param5, void *param6, void *param7, void *param8)
{
	void *p = (void*) par_func;
	void (*par_func_8param)(void*,void*,void*,void*,void*,void*,void*,void*,void*,void*) =  (void(*)(void*,void*,void*,void*,void*,void*,void*,void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_8param (p1, p2, param1, param2, param3, param4, param5, param6, param7, param8);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_7param (int *p1, int *p2, void *param1, void *param2, void *param3, void *param4, void *param5, void *param6, void *param7)
{
	void *p = (void*) par_func;
	void (*par_func_7param)(void*,void*,void*,void*,void*,void*,void*,void*,void*) =  (void(*)(void*,void*,void*,void*,void*,void*,void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_7param (p1, p2, param1, param2, param3, param4, param5, param6, param7);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_6param (int *p1, int *p2, void *param1, void *param2, void *param3, void *param4, void *param5, void *param6)
{
	void *p = (void*) par_func;
	void (*par_func_6param)(void*,void*,void*,void*,void*,void*,void*,void*) =  (void(*)(void*,void*,void*,void*,void*,void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_6param (p1, p2, param1, param2, param3, param4, param5, param6);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_5param (int *p1, int *p2, void *param1, void *param2, void *param3, void *param4, void *param5)
{
	void *p = (void*) par_func;
	void (*par_func_5param)(void*,void*,void*,void*,void*,void*,void*) =  (void(*)(void*,void*,void*,void*,void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_5param (p1, p2, param1, param2, param3, param4, param5);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_4param (int *p1, int *p2, void *param1, void *param2, void *param3, void *param4)
{
	void *p = (void*) par_func;
	void (*par_func_4param)(void*,void*,void*,void*,void*,void*) =  (void(*)(void*,void*,void*,void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_4param (p1, p2, param1, param2, param3, param4);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_3param (int *p1, int *p2, void *param1, void *param2, void *param3)
{
	void *p = (void*) par_func;
	void (*par_func_3param)(void*,void*,void*,void*,void*) =  (void(*)(void*,void*,void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_3param (p1, p2, param1, param2, param3);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_2param (int *p1, int *p2, void *param1, void *param2)
{
	void *p = (void*) par_func;
	void (*par_func_2param)(void*,void*,void*,void*) =  (void(*)(void*,void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_2param (p1, p2, param1, param2);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_1param (int *p1, int *p2, void *param1)
{
	void *p = (void*) par_func;
	void (*par_func_1param)(void*,void*,void*) =  (void(*)(void*,void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_1param (p1, p2, param1);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

static void __kmpc_parallel_func_0param (int *p1, int *p2)
{
	void *p = (void*) par_func;
	void (*par_func_0param)(void*,void*) =  (void(*)(void*,void*)) par_func;

	Backend_Enter_Instrumentation (1);
	Probe_OpenMP_UF_Entry ((UINT64) p);
	par_func_0param (p1, p2);
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

void __kmpc_fork_call (void *p1, int p2, void *p3, ...)
{
	void *params[8];
	va_list ap;
	int i;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": __kmpc_fork_call is at %p\n", __kmpc_fork_call_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_fork_call params %p %d %p\n", p1, p2, p3);
#endif

	if (__kmpc_fork_call_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
		Probe_OpenMP_ParDO_Entry ();

		/* Grab parameters */
		va_start (ap, p3);
		for (i = 0; i < p2; i++)
			params[i] = va_arg (ap, void*);
		va_end (ap);

		par_func = p3;

		switch (p2)
		{
			case 8:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_8param, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
				break;
			case 7:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_7param, params[0], params[1], params[2], params[3], params[4], params[5], params[6]);
				break;
			case 6:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_6param, params[0], params[1], params[2], params[3], params[4], params[5]);
				break;
			case 5:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_5param, params[0], params[1], params[2], params[3], params[4]);
				break;
			case 4:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_4param, params[0], params[1], params[2], params[3]);
				break;
			case 3:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_3param, params[0], params[1], params[2]);
				break;
			case 2:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_2param, params[0], params[1]);
				break;
			case 1:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_1param, params[0]);
				break;
			case 0:
				__kmpc_fork_call_real (p1, p2, __kmpc_parallel_func_0param);
				break;
			default:
				fprintf (stderr, PACKAGE_NAME": Error! Unhandled __kmpc_fork_call with %d arguments! Quitting!\n", p2);
				exit (-1);
				break;
		}

		Probe_OpenMP_ParDO_Exit ();	
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
	fprintf (stderr, PACKAGE_NAME": __kmpc_barrier is at %p\n", __kmpc_barrier_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_barrier params %p %d\n", p1, p2);
#endif

	if (__kmpc_barrier_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
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
	fprintf (stderr, PACKAGE_NAME": __kmpc_critical is at %p\n", __kmpc_critical_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_critical params %p %d %p\n", p1, p2, p3);
#endif

	if (__kmpc_critical_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
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
	fprintf (stderr, PACKAGE_NAME": __kmpc_end_critical is at %p\n", __kmpc_end_critical_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_end_critical params %p %d %p\n", p1, p2, p3);
#endif

	if (__kmpc_end_critical_real != NULL)
	{
		Backend_Enter_Instrumentation (1);
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
	fprintf (stderr, PACKAGE_NAME": __kmpc_dispatch_next_4 is at %p\n", __kmpc_dispatch_next_4_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_dispatch_next_4 params %p %d %p %p %p %p\n", p1, p2, p3, p4, p5, p6);
#endif

	if (__kmpc_dispatch_next_8_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_4_real (p1, p2, p3, p4, p5, p6);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": __kmpc_dispatch_next_8 is at %p\n", __kmpc_dispatch_next_8_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_dispatch_next_8 params %p %d %p %p %p %p\n", p1, p2, p3, p4, p5, p6);
#endif
	if (__kmpc_dispatch_next_8_real != NULL)
	{
		Backend_Enter_Instrumentation (2);
		Probe_OpenMP_Work_Entry();
		res = __kmpc_dispatch_next_8_real (p1, p2, p3, p4, p5, p6);
		Probe_OpenMP_Work_Exit();
		Backend_Leave_Instrumentation ();
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
	fprintf (stderr, PACKAGE_NAME": __kmpc_single is at %p\n", __kmpc_single_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_single params %p %d\n", p1, p2);
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
	fprintf (stderr, PACKAGE_NAME": __kmpc_end_single is at %p\n", __kmpc_single_real);
	fprintf (stderr, PACKAGE_NAME": __kmpc_end_single params %p %d\n", p1, p2);
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
