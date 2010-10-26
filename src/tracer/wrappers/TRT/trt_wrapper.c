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

#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef  __USE_GNU
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "wrapper.h"
#include "trace_macros.h"
#include "trt_probe.h"

#define DEBUG

static void* (*threadSpawn_real)(void *fun, void *arg) = NULL;
static void* (*threadRead_real)(void*) = NULL;

static void GetTRTHookPoints (int rank)
{
	/* Obtain @ for pthread_create */
	threadSpawn_real = (void*(*)(void*, void*)) dlsym (RTLD_NEXT, "threadSpawn");
	if (threadSpawn_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find threadSpawn in DSOs!!\n");

	/* Obtain @ for pthread_join */
	threadRead_real = (void*(*)(void*)) dlsym (RTLD_NEXT, "threadRead");
	if (threadRead_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find threadRead in DSOs!!\n");
}

/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	 INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/
struct thread_create_info
{
	void *(*routine)(void*);
	void *arg;
	
	pthread_cond_t wait;
	pthread_mutex_t lock;
};

static void * threadSpawn_hook (void *p1)
{
	struct thread_create_info *i = (struct thread_create_info*)p1;
	void *(*routine)(void*) = i->routine;
	void *arg = i->arg;
	void *res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG threadSpawn_hook (p1=%p)\n", p1);
#endif

	/* Notify the calling thread */
	pthread_mutex_lock (&(i->lock));
	pthread_cond_signal (&(i->wait));
	pthread_mutex_unlock (&(i->lock));

	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADFUNC_EV, (UINT64) routine, EMPTY);
	res = routine (arg);
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADFUNC_EV, EVT_END ,EMPTY);

	return res;
}

void * threadSpawn (void *p1, void *p2)
{
	void *res;
	struct thread_create_info i;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG threadSpawn (p1=%p, p2=%p)\n", p1, p2);
	fprintf (stderr, PACKAGE_NAME": DEBUG active threads in the backend = %d\n", Backend_getNumberOfThreads()+1);
#endif

	Probe_threadSpawn_Entry (p1);
		
	pthread_cond_init (&(i.wait), NULL);
	pthread_mutex_init (&(i.lock), NULL);
	pthread_mutex_lock (&(i.lock));

	i.arg = p2;
	i.routine = p1;

	Backend_ChangeNumberOfThreads (Backend_getNumberOfThreads()+1);

	res = threadSpawn_real (threadSpawn_hook, (void*) &i);

	/* if (0 == res) */
		/* if succeded, wait for a completion on copy the info */
		pthread_cond_wait (&(i.wait), &(i.lock));

	pthread_mutex_unlock (&(i.lock));
	pthread_mutex_destroy (&(i.lock));
	pthread_cond_destroy (&(i.wait));

	Probe_threadSpawn_Exit ();

	return res;
}

void * threadRead (void *p1)
{
	void *res;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG threadRead (p1=%p)\n", p1);
#endif

	Probe_threadRead_Entry ();
	res = threadRead_real (p1);
	Probe_threadRead_Exit ();
	return res;
}

/*
  This __attribute__ tells the loader to run this routine when
  the shared library is loaded 
*/
void __attribute__ ((constructor)) TRT_tracing_init(void);
void TRT_tracing_init (void)
{
	GetTRTHookPoints (0);
}

