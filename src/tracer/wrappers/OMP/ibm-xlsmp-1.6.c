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

#include "wrapper.h"
#include "omp-common.h"

#if defined(PIC)

//#define DEBUG

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

typedef unsigned long long ull;
typedef long long ll;

/* Pointer to the user function called by a PARALLEL DO REGION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*pardo_uf)(char*, long, long, unsigned);

/* Pointer to the user function called by a DO REGION */
static void (*do_uf[MAX_THD])(char*, long, long) = {NULL};

/* Pointer to the user function called by a PARALLEL REGION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*par_uf)(char*);

/* Pointer to the user function called by a SINGLE REGION */
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (*par_single)(void);

/* Pointer to the user function called by a SECTIONS */
typedef void (**_xlsmp_sections)(char*,unsigned);
static _xlsmp_sections real_sections[MAX_THD];
long num_real_sections[MAX_THD];

/*
		callme_pardo (char*, ull, ull)
		With the same header as the routine to be called by the SMP runtime, just
    acts as a trampoline to this call. Invokes the required iterations of the
    parallel do loop.
*/
static void callme_pardo (char *ptr, long lbnd, long ubnd, unsigned thid)
{
	void *p = *((void**) pardo_uf);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": callme_pardo: ptr=%p lbnd=%ld ubnd=%ld thid=%u\n", ptr, lbnd, ubnd, thid);
	fprintf (stderr, PACKAGE_NAME": callme_pardo: pardo_uf=%p\n", p);
#endif

	Extrae_OpenMP_UF_Entry (p);
	Backend_Leave_Instrumentation (); /* We're entering in user code */
	pardo_uf (ptr, lbnd, ubnd, thid);
	Extrae_OpenMP_UF_Exit ();
}

/*
	callme_do (char*, ull, ull)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Invokes the required iterations of the
	parallel do loop.
*/
static void callme_do (char *ptr, long lbnd, long ubnd)
{
	void *p = *((void**) do_uf[THREADID]);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": callme_do: ptr=%p lbnd=%ld ubnd=%ld\n", ptr, lbnd, ubnd);
	fprintf (stderr, PACKAGE_NAME": callme_do: do_uf=%p\n", p);
#endif

	Extrae_OpenMP_UF_Entry (p);
	Backend_Leave_Instrumentation (); /* We're entering in user code */
	do_uf[THREADID] (ptr, lbnd, ubnd);
	Extrae_OpenMP_UF_Exit ();
}
/*
	callme_par (char*)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Each thread runs the very same routine
	with different params.
*/
static void callme_par (char *ptr)
{
	void *p = *((void**) par_uf);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": callme_par: ptr=%p\n", ptr);
	fprintf (stderr, PACKAGE_NAME": callme_par: par_uf=%p\n", p);
#endif

	Extrae_OpenMP_UF_Entry (p);
	Backend_Leave_Instrumentation (); /* We're entering in user code */
	par_uf (ptr);
	Extrae_OpenMP_UF_Exit ();
}

/*
	callme_single (void)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Each thread runs the very same routine
	with different params.
*/
static void callme_single(void)
{
	void *p = *((void**) par_single);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": callme_single: par_single=%p\n", p);
#endif

	Extrae_OpenMP_UF_Entry (p);
	Backend_Leave_Instrumentation (); /* We're entering in user code */
	par_single ();
	Extrae_OpenMP_UF_Exit ();
}

/*
	callme_section (void)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Each thread runs the very same routine
	with different params.
*/
static volatile long long __atomic_index;
#if !defined(HAVE__SYNC_FETCH_AND_ADD)
static pthread_mutex_t __atomic_index_mtx = PTHREAD_MUTEX_INITIALIZER;
#endif

static void callme_section(char *p1, unsigned p2)
{
	long long index;
#if defined(HAVE__SYNC_FETCH_AND_ADD)
	index = __sync_fetch_and_add(&__atomic_index,1);
#else
	pthread_mutex_lock (&__atomic_index_mtx);
	index = __atomic_index;
	__atomic_index++;
	pthread_mutex_unlock (&__atomic_index_mtx);
#endif

	if (index < num_real_sections[THREADID])
	{
		_xlsmp_sections real = real_sections[THREADID];

		Extrae_OpenMP_UF_Entry (real[index]);
		Backend_Leave_Instrumentation (); /* We're entering in user code */
		real[index] (p1, p2);
		Extrae_OpenMP_UF_Exit ();
	}
}

static void (*_xlsmpParallelDoSetup_TPO_real)(int,void**,long,long,long,long,void**,void**,void**,long,long,void**,long) = NULL;
static void (*_xlsmpParRegionSetup_TPO_real)(int,void*,int,void*,void*,void**,long,long) = NULL;
static void (*_xlsmpWSDoSetup_TPO_real)(int,void*,long,long,long,long,void*,void*,void**,long) = NULL;
static void (*_xlsmpSingleSetup_TPO_real)(int,void*,int,void*) = NULL;
static void (*_xlsmpWSSectSetup_TPO_real)(int,void*,long,void*,void*,void**,long,long) = NULL;
static void (*_xlsmpBarrier_TPO_real)(int,int*) = NULL;
static void (*_xlsmpGetDefaultSLock_real)(void*) = NULL;
static void (*_xlsmpRelDefaultSLock_real)(void*) = NULL;
static void (*_xlsmpGetSLock_real)(void*) = NULL;
static void (*_xlsmpRelSLock_real)(void*) = NULL;

extern int omp_get_max_threads();

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

static int ibm_xlsmp_1_6_GetOpenMPHookPoints (int rank)
{
	int count = 0;

	UNREFERENCED_PARAMETER(rank)

	/* Obtain @ for _xlsmpParallelDoSetup_TPO */
	_xlsmpParallelDoSetup_TPO_real =
		(void(*)(int,void**,long,long,long,long,void**,void**,void**,long,long,void**,long))
		dlsym (RTLD_NEXT, "_xlsmpParallelDoSetup_TPO");
	INC_IF_NOT_NULL(_xlsmpParallelDoSetup_TPO_real,count);

	/* Obtain @ for _xlsmpParRegionSetup_TPO */
	_xlsmpParRegionSetup_TPO_real =
		(void(*)(int,void*,int,void*,void*,void**,long,long))
		dlsym (RTLD_NEXT, "_xlsmpParRegionSetup_TPO");
	INC_IF_NOT_NULL(_xlsmpParRegionSetup_TPO_real,count);

	/* Obtain @ for _xlsmpWSDoSetup_TPO */
	_xlsmpWSDoSetup_TPO_real =
		(void(*)(int,void*,long,long,long,long,void*,void*,void**,long))
		dlsym (RTLD_NEXT, "_xlsmpWSDoSetup_TPO");
	INC_IF_NOT_NULL(_xlsmpWSDoSetup_TPO_real,count);

	/* Obtain @ for _xlsmpWSSectSetup_TPO */
	_xlsmpWSSectSetup_TPO_real =
		(void(*)(int,void*,long,void*,void*,void**,long,long))
		dlsym (RTLD_NEXT, "_xlsmpWSSectSetup_TPO");
	INC_IF_NOT_NULL(_xlsmpWSSectSetup_TPO_real,count);

	/* Obtain @ for _xlsmpSingleSetup_TPO */
	_xlsmpSingleSetup_TPO_real =
		(void(*)(int,void*,int,void*)) dlsym (RTLD_NEXT, "_xlsmpSingleSetup_TPO");
	INC_IF_NOT_NULL(_xlsmpSingleSetup_TPO_real,count);

	/* Obtain @ for _xlsmpBarrier_TPO */
	_xlsmpBarrier_TPO_real =
		(void(*)(int,int*)) dlsym (RTLD_NEXT, "_xlsmpBarrier_TPO");
	INC_IF_NOT_NULL(_xlsmpBarrier_TPO_real,count);

	/* Obtain @ for _xlsmpGetDefaultSLock */
	_xlsmpGetDefaultSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpGetDefaultSLock");
	INC_IF_NOT_NULL(_xlsmpGetDefaultSLock_real,count);

	/* Obtain @ for _xlsmpRelDefaultSLock */
	_xlsmpRelDefaultSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpRelDefaultSLock");
	INC_IF_NOT_NULL(_xlsmpRelDefaultSLock_real,count);

	/* Obtain @ for _xlsmpGetSLock */
	_xlsmpGetSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpGetSLock");
	INC_IF_NOT_NULL(_xlsmpGetSLock_real,count);

	/* Obtain @ for _xlsmpRelSLock */
	_xlsmpRelSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpRelSLock");
	INC_IF_NOT_NULL(_xlsmpRelSLock_real,count);

	/* Any hook point? */
	return count > 0;
}

/*
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
*/

void _xlsmpParallelDoSetup_TPO(int p1, void *p2, long p3, long p4, long p5, long p6, void *p7, void *p8, void **p9, long p10, long p11, void *p12, long p13)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpParallelDoSetup_TPO is at %p\n", _xlsmpParallelDoSetup_TPO_real);
	fprintf (stderr, PACKAGE_NAME": _xlsmpParallelDoSetup_TPO are %d %p %ld %ld %ld %ld %p %p %p %ld %ld %p %ld\n", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
#endif

	if (_xlsmpParallelDoSetup_TPO_real != NULL && mpitrace_on)
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(char*, long, long, unsigned))p2;

		Extrae_OpenMP_ParDO_Entry ();
		_xlsmpParallelDoSetup_TPO_real (p1, (void**)callme_pardo, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
		Extrae_OpenMP_ParDO_Exit ();	
	}
	else if (_xlsmpParallelDoSetup_TPO_real != NULL && !mpitrace_on)
	{
		_xlsmpParallelDoSetup_TPO_real (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpParallelDoSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpParRegionSetup_TPO (int p1, void *p2, int p3, void* p4, void* p5, void** p6, long p7, long p8)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpParRegionSetup_TPO is at %p\n", _xlsmpParRegionSetup_TPO_real);
	fprintf (stderr, PACKAGE_NAME": _xlsmpParRegionSetup_TPO params %d %p %d %p %p %p %ld %ld\n", p1, p2, p3, p4, p5, p6, p7, p8);
#endif

	if (_xlsmpParRegionSetup_TPO_real != NULL && mpitrace_on)
	{
		/* Set the pointer to the correct PARALLEL user function */
		par_uf = (void(*)(char*))p2;

		/* Reset the counter of the sections to 0 */
		__atomic_index = 0;

		Extrae_OpenMP_ParRegion_Entry();
		_xlsmpParRegionSetup_TPO_real (p1, callme_par, p3, p4, p5, p6, p7, p8);
		Extrae_OpenMP_ParRegion_Exit();
  }
	else if (_xlsmpParRegionSetup_TPO_real != NULL && !mpitrace_on)
	{
		_xlsmpParRegionSetup_TPO_real (p1, p2, p3, p4, p5, p6, p7, p8);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpParRegionSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpWSDoSetup_TPO (int p1, void *p2, long p3, long p4, long p5, long p6, void* p7, void* p8, void** p9, long p10)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpWSDoSetup_TPO is at %p\n", _xlsmpWSDoSetup_TPO_real);
	fprintf (stderr, PACKAGE_NAME": _xlsmpWSDoSetup_TPO params %d %p %ld %ld %ld %ld %p %p %p %ld\n", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
#endif

	if (_xlsmpWSDoSetup_TPO_real != NULL && mpitrace_on)
	{
		/* Set the pointer to the correct DO user function */
		do_uf[THREADID] = (void(*)(char*, long, long))p2;

		Extrae_OpenMP_DO_Entry ();
		_xlsmpWSDoSetup_TPO_real
			(p1, callme_do, p3, p4, p5, p6, p7, p8, p9, p10);
		Extrae_OpenMP_DO_Exit();
	}
	else if (_xlsmpWSDoSetup_TPO_real != NULL && !mpitrace_on)
	{
		_xlsmpWSDoSetup_TPO_real (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpWSDoSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpBarrier_TPO (int p1, int *p2)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpBarrier_TPO is at %p\n", _xlsmpBarrier_TPO_real);
	fprintf (stderr, PACKAGE_NAME": _xlsmpBarrier_TPO params %d %p\n", p1, p2);
#endif

	if (_xlsmpBarrier_TPO_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Barrier_Entry ();
		_xlsmpBarrier_TPO_real (p1, p2);
		Extrae_OpenMP_Barrier_Exit ();
	}
	else if (_xlsmpBarrier_TPO_real != NULL && !mpitrace_on)
	{
		_xlsmpBarrier_TPO_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpBarrier_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}
 
void _xlsmpSingleSetup_TPO (int p1, void *p2, int p3, void *p4)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpSingleSetup_TPO is at %p\n", _xlsmpBarrier_TPO_real);
	fprintf (stderr, PACKAGE_NAME": _xlsmpSingleSetup_TPO params %d %p %d %p\n", p1, p2, p3, p4);
#endif

	if (_xlsmpSingleSetup_TPO_real != NULL && mpitrace_on)
	{
		/* Set the pointer to the correct SINGLE user function */
		par_single = (void(*)(void))p2;

		Extrae_OpenMP_Single_Entry();
		_xlsmpSingleSetup_TPO_real (p1, callme_single, p3, p4);
		Extrae_OpenMP_Single_Exit();
	}
	else if (_xlsmpSingleSetup_TPO_real != NULL && !mpitrace_on)
	{
		_xlsmpSingleSetup_TPO_real (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpSingleSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpWSSectSetup_TPO (int p1, void *p2, long p3, void *p4, void *p5, void** p6, long p7, long p8)
{
	long index = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": THREADID %u _xlsmpWSSectSetup_TPO is at %p\n", THREADID, _xlsmpWSSectSetup_TPO_real);
	fprintf (stderr, PACKAGE_NAME": THREADID %u _xlsmpWSSectSetup_TPO params %d %p %ld %p %p %p %ld %ld\n", THREADID, p1, p2, p3, p4, p5, p6, p7, p8);
#endif

	if (_xlsmpWSSectSetup_TPO_real != NULL && mpitrace_on)
	{
		/* Just intercept the @ of the routines representing all the sections to
		   call our routine and run them from inside! ( see callme_section ) */

		void (*callme_sections[p3])(char*,unsigned);
		for (index = 0; index < p3; index++)
			callme_sections[index] = callme_section;

		real_sections[THREADID] = (_xlsmp_sections) p2;
		num_real_sections[THREADID] = p3;

		Extrae_OpenMP_Section_Entry();
		_xlsmpWSSectSetup_TPO_real (p1, callme_sections, p3, p4, p5, p6, p7, p8);
		Extrae_OpenMP_Section_Exit();
	}
	else if (_xlsmpWSSectSetup_TPO_real != NULL && !mpitrace_on)
	{
		_xlsmpWSSectSetup_TPO_real (p1, p2, p3, p4, p5, p6, p7, p8);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpWSSectSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpRelDefaultSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpRelDefaultSLock is at %p\n", _xlsmpRelDefaultSLock_real);
#endif

	if (_xlsmpRelDefaultSLock_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Unnamed_Unlock_Entry();
		_xlsmpRelDefaultSLock_real(p1);
		Extrae_OpenMP_Unnamed_Unlock_Exit();
	}
	else if (_xlsmpRelDefaultSLock_real != NULL && !mpitrace_on)
	{
		_xlsmpRelDefaultSLock_real(p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpRelDefaultSLock is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpRelSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpRelSLock is at %p\n", _xlsmpRelSLock_real);
#endif

	if (_xlsmpRelSLock_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Named_Unlock_Entry(p1);
		_xlsmpRelSLock_real(p1);
		Extrae_OpenMP_Named_Unlock_Exit();
	}
	else if (_xlsmpRelSLock_real != NULL && mpitrace_on)
	{
		_xlsmpRelSLock_real(p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpRelSLock is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpGetDefaultSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpGetDefaultSLock is at %p\n", _xlsmpGetDefaultSLock_real);
#endif

	if (_xlsmpGetDefaultSLock_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Unnamed_Lock_Entry();
		_xlsmpGetDefaultSLock_real (p1);
		Extrae_OpenMP_Unnamed_Lock_Exit();
	}
	else if (_xlsmpGetDefaultSLock_real != NULL && !mpitrace_on)
	{
		_xlsmpGetDefaultSLock_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpGetDefaultSLock is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpGetSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": _xlsmpGetSLock is at %p\n", _xlsmpGetSLock_real);
#endif

	if (_xlsmpGetSLock_real != NULL && mpitrace_on)
	{
		Extrae_OpenMP_Named_Lock_Entry();
		_xlsmpGetSLock_real (p1);
		Extrae_OpenMP_Named_Lock_Exit(p1);
	}
	else if (_xlsmpGetSLock_real != NULL && !mpitrace_on)
	{
		_xlsmpGetSLock_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": _xlsmpGetSLock is not hooked! exiting!!\n");
		exit (0);
	}
}

extern int omp_get_max_threads();

int ibm_xlsmp_1_6_hook_points (int ntask)
{
	int hooked;
	int max_threads;

	hooked = ibm_xlsmp_1_6_GetOpenMPHookPoints (ntask);
   
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

#endif /* PIC */
