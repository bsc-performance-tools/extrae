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
#include "pthread_probe.h"

static int (*pthread_create_real)(pthread_t*,const pthread_attr_t*,void *(*) (void *),void*) = NULL;
static int (*pthread_join_real)(pthread_t,void**) = NULL;
static int (*pthread_detach_real)(pthread_t) = NULL;

static void GetpthreadHookPoints (int rank)
{
	/* Obtain @ for pthread_create */
	pthread_create_real =
		(int(*)(pthread_t*,const pthread_attr_t*,void *(*) (void *),void*))
		dlsym (RTLD_NEXT, "pthread_create");
	if (pthread_create_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_create in DSOs!!\n");

	/* Obtain @ for pthread_join */
	pthread_join_real =
		(int(*)(pthread_t,void**)) dlsym (RTLD_NEXT, "pthread_join");
	if (pthread_join_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_join in DSOs!!\n");

	/* Obtain @ for pthread_detach */
	pthread_detach_real = (int(*)(pthread_t)) dlsym (RTLD_NEXT, "pthread_detach");
	if (pthread_detach_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_detach in DSOs!!\n");
}

/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	 INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/
struct pthread_create_info
{
	int pthreadID;
	void *(*routine)(void*);
	void *arg;
	
	pthread_cond_t wait;
	pthread_mutex_t lock;
};

static void * pthread_create_hook (void *p1)
{
	struct pthread_create_info *i = (struct pthread_create_info*)p1;
	void *(*routine)(void*) = i->routine;
	void *arg = i->arg;
	void *res;

	Backend_SetpThreadIdentifier (i->pthreadID);

	/* incialitzar hwc, mode (detail/burst), primers events (tempsinit?) */

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

int pthread_create (pthread_t* p1, const pthread_attr_t* p2,
	void *(*p3) (void *), void* p4)
{
	static int pthread_library_depth = 0;
	int res;
	struct pthread_create_info i;

	/* This is a bit tricky.
	   Some OSes (like FreeBSD) delegates the pthread library initialization
	   to the very first call of pthread_create. In order to initialize the 
	   library the OS calls pthread_create again to create the structure for
	   the main thread.
	   So, pthread_library_depth > 0 controls this situation
	*/

	if (0 == pthread_library_depth)
	{
		pthread_library_depth++;

		Probe_pthread_Create_Entry (p3);
		
		pthread_cond_init (&(i.wait), NULL);
		pthread_mutex_init (&(i.lock), NULL);
		pthread_mutex_lock (&(i.lock));

		i.arg = p4;
		i.routine = p3;
		i.pthreadID = Backend_getNumberOfThreads();

		Backend_ChangeNumberOfThreads (i.pthreadID+1);

		res = pthread_create_real (p1, p2, pthread_create_hook, (void*) &i);

		if (0 == res)
			/* if succeded, wait for a completion on copy the info */
			pthread_cond_wait (&(i.wait), &(i.lock));

		pthread_mutex_unlock (&(i.lock));
		pthread_mutex_destroy (&(i.lock));
		pthread_cond_destroy (&(i.wait));

		Probe_pthread_Create_Exit ();

		pthread_library_depth--;
	}
	else
		res = pthread_create_real (p1, p2, p3, p4);

	return res;
}

int pthread_join (pthread_t p1, void **p2)
{
	int res;
	Probe_pthread_Join_Entry ();
	res = pthread_join_real (p1, p2);
	Probe_pthread_Join_Exit ();
	return res;
}

int pthread_detach (pthread_t p1)
{
	int res;
	Probe_pthread_Detach_Entry ();
	res = pthread_detach_real (p1);
	Probe_pthread_Detach_Exit ();
	return res;
}

/*
  This __attribute__ tells the loader to run this routine when
  the shared library is loaded 
*/
void __attribute__ ((constructor)) pthread_tracing_init(void);
void pthread_tracing_init (void)
{
	GetpthreadHookPoints (0);
	Backend_CreatepThreadIdentifier ();
}
