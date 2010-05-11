/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
/* FIXME: Array of function pointers indexed by thread? (nowait issue) */
static void (**par_sections)(void);

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
	fprintf (stderr, "mpitrace: callme_pardo: ptr=%p lbnd=%ld ubnd=%ld thid=%u\n", ptr, lbnd, ubnd, thid);
	fprintf (stderr, "mpitrace: callme_pardo: pardo_uf=%p\n", p);
#endif

	Probe_OpenMP_UF ((UINT64) p);
	pardo_uf (ptr, lbnd, ubnd, thid);
	Probe_OpenMP_UF (EVT_END);
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
	fprintf (stderr, "mpitrace: callme_do: ptr=%p lbnd=%ld ubnd=%ld\n", ptr, lbnd, ubnd);
	fprintf (stderr, "mpitrace: callme_do: do_uf=%p\n", p);
#endif

	Probe_OpenMP_UF ((UINT64) p);
	do_uf[THREADID] (ptr, lbnd, ubnd);
	Probe_OpenMP_UF (EVT_END);
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
	fprintf (stderr, "mpitrace: callme_par: ptr=%p\n", ptr);
	fprintf (stderr, "mpitrace: callme_par: par_uf=%p\n", p);
#endif

	Probe_OpenMP_UF ((UINT64) p);
	par_uf (ptr);
	Probe_OpenMP_UF (EVT_END);
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
	fprintf (stderr, "mpitrace: callme_single: par_single=%p\n", p);
#endif

	Probe_OpenMP_UF ((UINT64) p);
	par_single ();
	Probe_OpenMP_UF (EVT_END);
}

/*
	callme_section (void)
	With the same header as the routine to be called by the SMP runtime, just
	acts as a trampoline to this call. Each thread runs the very same routine
	with different params.
*/

#if defined(OS_LINUX) && defined(ARCH_PPC)
# ifdef HAVE_ARCH_POWERPC_INCLUDE_ASM_ATOMIC_H
#  include <arch/powerpc/include/asm/atomic.h>
# else
#  if SIZEOF_VOIDP == 8
#   ifdef HAVE_ASM_PPC64_ATOMIC_H
#    define __KERNEL__  /* patch to workaround an #ifdef inside atomic.h */
#    include <asm-ppc64/atomic.h>
#    undef  __KERNEL__
#   endif
#  elif SIZEOF_VOIDP == 4
#   ifdef HAVE_ASM_PPC_ATOMIC_H
#    define __KERNEL__  /* patch to workaround an #ifdef inside atomic.h */
#    include <asm-ppc/atomic.h>
#    undef  __KERNEL__
#   endif
#  else
#   error "Unknown memory model!"
#  endif
# endif
#else
# error "This file can only be compiled at linux/ppc nowadays!"
#endif

static atomic_t atomic_index;

static void callme_section(void)
{
	int index = atomic_inc_return (&atomic_index)-1;

#if defined(DEBUG)
	fprintf (stderr, "mpitrace: callme_section: par_sections[%d]=%p\n", index, par_sections[index-1]);
#endif

	Probe_OpenMP_UF ((UINT64) par_sections[index]);
	par_sections[index]();
	Probe_OpenMP_UF (EVT_END);
}

static void (*_xlsmpParallelDoSetup_TPO_real)(int,void**,long,long,long,long,void**,void**,void**,long,long,void**,long) = NULL;
static void (*_xlsmpParRegionSetup_TPO_real)(int,void*,int,void*,void*,void**,long,long) = NULL;
static void (*_xlsmpWSDoSetup_TPO_real)(int,void*,long,long,long,long,void*,void*,void**,long) = NULL;
static void (*_xlsmpSingleSetup_TPO_real)(int,void*,int,void*,void*,int) = NULL;
static void (*_xlsmpWSSectSetup_TPO_real)(int,void*,long,void*,void*,void**,long,long) = NULL;
static void (*_xlsmpBarrier_TPO_real)(int,int*) = NULL;
static void (*_xlsmpGetDefaultSLock_real)(void*) = NULL;
static void (*_xlsmpRelDefaultSLock_real)(void*) = NULL;
static void (*_xlsmpGetSLock_real)(void*) = NULL;
static void (*_xlsmpRelSLock_real)(void*) = NULL;

#define INC_IF_NOT_NULL(ptr,cnt) (cnt = (ptr == NULL)?cnt:cnt+1)

static int ibm_xlsmp_1_6_GetOpenMPHookPoints(int rank)
{
	int count = 0;

	/* Obtain @ for _xlsmpParallelDoSetup_TPO */
	_xlsmpParallelDoSetup_TPO_real =
		(void(*)(int,void**,long,long,long,long,void**,void**,void**,long,long,void**,long))
		dlsym (RTLD_NEXT, "_xlsmpParallelDoSetup_TPO");
	if (_xlsmpParallelDoSetup_TPO_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpParallelDoSetup_TPO in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpParallelDoSetup_TPO_real,count);

	/* Obtain @ for _xlsmpParRegionSetup_TPO */
	_xlsmpParRegionSetup_TPO_real =
		(void(*)(int,void*,int,void*,void*,void**,long,long))
		dlsym (RTLD_NEXT, "_xlsmpParRegionSetup_TPO");
	if (_xlsmpParRegionSetup_TPO_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpParRegionSetup_TPO in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpParRegionSetup_TPO_real,count);

	/* Obtain @ for _xlsmpWSDoSetup_TPO */
	_xlsmpWSDoSetup_TPO_real =
		(void(*)(int,void*,long,long,long,long,void*,void*,void**,long))
		dlsym (RTLD_NEXT, "_xlsmpWSDoSetup_TPO");
	if (_xlsmpWSDoSetup_TPO_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpWSDoSetup_TPO in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpWSDoSetup_TPO_real,count);

	/* Obtain @ for _xlsmpWSSectSetup_TPO */
	_xlsmpWSSectSetup_TPO_real =
		(void(*)(int,void*,long,void*,void*,void**,long,long))
		dlsym (RTLD_NEXT, "_xlsmpWSSectSetup_TPO");
	if (_xlsmpWSSectSetup_TPO_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpWSSectSetup_TPO in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpWSSectSetup_TPO_real,count);

	/* Obtain @ for _xlsmpSingleSetup_TPO */
	_xlsmpSingleSetup_TPO_real =
		(void(*)(int,void*,int,void*,void*,int)) dlsym (RTLD_NEXT, "_xlsmpSingleSetup_TPO");
	if (_xlsmpSingleSetup_TPO_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpSingleSetup_TPO in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpSingleSetup_TPO_real,count);

	/* Obtain @ for _xlsmpBarrier_TPO */
	_xlsmpBarrier_TPO_real =
		(void(*)(int,int*)) dlsym (RTLD_NEXT, "_xlsmpBarrier_TPO");
	if (_xlsmpBarrier_TPO_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpBarrier_TPO in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpBarrier_TPO_real,count);

	/* Obtain @ for _xlsmpGetDefaultSLock */
	_xlsmpGetDefaultSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpGetDefaultSLock");
	if (_xlsmpGetDefaultSLock_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpGetDefaultSLock in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpGetDefaultSLock_real,count);

	/* Obtain @ for _xlsmpRelDefaultSLock */
	_xlsmpRelDefaultSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpRelDefaultSLock");
	if (_xlsmpRelDefaultSLock_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpRelDefaultSLock in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpRelDefaultSLock_real,count);

	/* Obtain @ for _xlsmpGetSLock */
	_xlsmpGetSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpGetSLock");
	if (_xlsmpGetSLock_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpGetSLock in DSOs!!\n");
	INC_IF_NOT_NULL(_xlsmpGetSLock_real,count);

	/* Obtain @ for _xlsmpRelSLock */
	_xlsmpRelSLock_real =
		(void(*)(void*)) dlsym (RTLD_NEXT, "_xlsmpRelSLock");
	if (_xlsmpRelSLock_real == NULL && rank == 0)
		fprintf (stderr, "mpitrace: Unable to find _xlsmpRelSLock in DSOs!!\n");
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
	fprintf (stderr, "mpitrace: _xlsmpParallelDoSetup_TPO is at %p\n", _xlsmpParallelDoSetup_TPO_real);
	fprintf (stderr, "mpitrace: p01 = %d\n", p1);
	fprintf (stderr, "mpitrace: p02 = %p (%p)\n", p2, (p2!=NULL)?*p2:NULL);
	fprintf (stderr, "mpitrace: p03 = %ld\n", p3);
	fprintf (stderr, "mpitrace: p04 = %ld\n", p4);
	fprintf (stderr, "mpitrace: p05 = %ld\n", p5);
	fprintf (stderr, "mpitrace: p06 = %ld\n", p6);
	fprintf (stderr, "mpitrace: p07 = %p (%p)\n", p7, (p7!=NULL)?*p7:NULL);
	fprintf (stderr, "mpitrace: p08 = %p (%p)\n", p8, (p8!=NULL)?*p8:NULL);
	fprintf (stderr, "mpitrace: p09 = %p (%p)\n", p9, (p9!=NULL)?*p9:NULL);
	fprintf (stderr, "mpitrace: p10 = %ld\n", p10);
	fprintf (stderr, "mpitrace: p11 = %ld\n", p11);
	fprintf (stderr, "mpitrace: p12 = %p (%p)\n", p12, (p12!=NULL)?*p12:NULL);
	fprintf (stderr, "mpitrace: p13 = %ld\n", p13);
#endif

	if (_xlsmpParallelDoSetup_TPO_real != NULL)
	{
		/* Set the pointer to the correct PARALLEL DO user function */
		pardo_uf = (void(*)(char*, long, long, unsigned))p2;

		Probe_OpenMP_ParDO_Entry ();
		_xlsmpParallelDoSetup_TPO_real (p1, (void**)callme_pardo, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
		Probe_OpenMP_ParDO_Exit ();	
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpParallelDoSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpParRegionSetup_TPO (int p1, void *p2, int p3, void* p4, void* p5, void** p6, long p7, long p8)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpParRegionSetup_TPO is at %p\n", _xlsmpParRegionSetup_TPO_real);
	fprintf (stderr, "mpitrace: _xlsmpParRegionSetup_TPO params %d %p %d %p %p %p %ld %ld\n", p1, p2, p3, p4, p5, p6, p7, p8);
#endif

	if (_xlsmpParRegionSetup_TPO_real != NULL)
	{
		/* Set the pointer to the correct PARALLEL user function */
		par_uf = (void(*)(char*))p2;

		Probe_OpenMP_ParRegion_Entry();
		_xlsmpParRegionSetup_TPO_real (p1, callme_par, p3, p4, p5, p6, p7, p8);
		Probe_OpenMP_ParRegion_Exit();
  }
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpParRegionSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpWSDoSetup_TPO (int p1, void *p2, long p3, long p4, long p5, long p6, void* p7, void* p8, void** p9, long p10)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpWSDoSetup_TPO is at %p\n", _xlsmpWSDoSetup_TPO_real);
	fprintf (stderr, "mpitrace: _xlsmpWSDoSetup_TPO params %d %p %ld %ld %ld %ld %p %p %p %ld\n", p1, p2, p3, p4, p5, p6, p7, p8, p9, p10);
#endif

	if (_xlsmpWSDoSetup_TPO_real != NULL)
	{
		/* Set the pointer to the correct DO user function */
		do_uf[THREADID] = (void(*)(char*, long, long))p2;

		Probe_OpenMP_DO_Entry ();
		_xlsmpWSDoSetup_TPO_real
			(p1, callme_do, p3, p4, p5, p6, p7, p8, p9, p10);
		Probe_OpenMP_DO_Exit();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpWSDoSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpBarrier_TPO (int p1, int *p2)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpBarrier_TPO is at %p\n", _xlsmpBarrier_TPO_real);
	fprintf (stderr, "mpitrace: _xlsmpBarrier_TPO params %d %p\n", p1, p2);
#endif

	if (_xlsmpBarrier_TPO_real != NULL)
	{
		Probe_OpenMP_Barrier_Entry ();
		_xlsmpBarrier_TPO_real (p1, p2);
		Probe_OpenMP_Barrier_Exit ();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpBarrier_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}
 
void _xlsmpSingleSetup_TPO (int p1, void *p2, int p3, void *p4, void *p5, int p6)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpSingleSetup_TPO is at %p\n", _xlsmpBarrier_TPO_real);
	fprintf (stderr, "mpitrace: _xlsmpSingleSetup_TPO params %d %p %d %p %p %d\n", p1, p2, p3, p4, p5, p6);
#endif

	if (_xlsmpSingleSetup_TPO_real != NULL)
	{
		/* Set the pointer to the correct SINGLE user function */
		par_single = (void(*)(void))p2;

		Probe_OpenMP_Single_Entry();
		_xlsmpSingleSetup_TPO_real (p1, callme_single, p3, p4, p5, p6);
		Probe_OpenMP_Single_Exit();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpSingleSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpWSSectSetup_TPO (int p1, void *p2, long p3, void *p4, void *p5, void** p6, long p7, long p8)
{
	ull index = 0;

#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpWSSectSetup_TPO is at %p\n", _xlsmpWSSectSetup_TPO_real);
	fprintf (stderr, "mpitrace: _xlsmpWSSectSetup_TPO params %d %p %ld %p %p %p %ld %ld\n", p1, p2, p3, p4, p5, p6, p7, p8);
#endif

	if (_xlsmpWSSectSetup_TPO_real != NULL)
	{
		/* Just intercept the @ of the routines representing all the sections to
		   call our routine and run them from inside! ( see callme_section ) */

		void (*callme_sections[p3])(void);
		for (index = 0; index < p3; index++)
			callme_sections[index] = callme_section;
		atomic_index.counter = 0;
		par_sections = (void(**)(void))p2;

		Probe_OpenMP_Section_Entry();
		_xlsmpWSSectSetup_TPO_real (p1, callme_sections, p3, p4, p5, p6, p7, p8);
		Probe_OpenMP_Section_Exit();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpWSSectSetup_TPO is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpRelDefaultSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpRelDefaultSLock is at %p\n", _xlsmpRelDefaultSLock_real);
#endif

	if (_xlsmpRelDefaultSLock_real != NULL)
	{
		Probe_OpenMP_Unnamed_Unlock_Entry();
		_xlsmpRelDefaultSLock_real(p1);
		Probe_OpenMP_Unnamed_Unlock_Exit();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpRelDefaultSLock is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpRelSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpRelSLock is at %p\n", _xlsmpRelSLock_real);
#endif

	if (_xlsmpRelSLock_real != NULL)
	{
		Probe_OpenMP_Named_Unlock_Entry();
		_xlsmpRelSLock_real(p1);
		Probe_OpenMP_Named_Unlock_Exit();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpRelSLock is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpGetDefaultSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpGetDefaultSLock is at %p\n", _xlsmpGetDefaultSLock_real);
#endif

	if (_xlsmpGetDefaultSLock_real != NULL)
	{
		Probe_OpenMP_Unnamed_Lock_Entry();
		_xlsmpGetDefaultSLock_real (p1);
		Probe_OpenMP_Unnamed_Lock_Exit();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpGetDefaultSLock is not hooked! exiting!!\n");
		exit (0);
	}
}

void _xlsmpGetSLock (void *p1)
{
#if defined(DEBUG)
	fprintf (stderr, "mpitrace: _xlsmpGetSLock is at %p\n", _xlsmpGetSLock_real);
#endif

	if (_xlsmpGetSLock_real != NULL)
	{
		Probe_OpenMP_Named_Lock_Entry();
		_xlsmpGetSLock_real (p1);
		Probe_OpenMP_Named_Lock_Exit();
	}
	else
	{
		fprintf (stderr, "mpitrace: _xlsmpGetSLock is not hooked! exiting!!\n");
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
		fprintf (stderr, "mpitrace: omp_get_max_threads() > MAX_THD. Aborting...\nRecompile MPItrace increasing MAX_THD or decrease OMP_NUM_THREADS\n");
		exit (1);
	}

	return hooked;
}

#endif /* !defined(DYNINST_MODULE) */
