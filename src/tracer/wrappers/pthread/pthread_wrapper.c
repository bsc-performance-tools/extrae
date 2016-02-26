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
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_TIME_H
# include <time.h>
#endif

#include "wrapper.h"
#include "trace_macros.h"
#include "pthread_probe.h"

// #define DEBUG
//#define DEBUG_MUTEX

#if defined(PIC)
static int (*pthread_create_real)(pthread_t*,const pthread_attr_t*,void *(*) (void *),void*) = NULL;
static int (*pthread_join_real)(pthread_t,void**) = NULL;
static int (*pthread_detach_real)(pthread_t) = NULL;
static void (*pthread_exit_real)(void*) = NULL;
#if defined(HAVE_PTHREAD_BARRIER_WAIT)
static int (*pthread_barrier_wait_real)(pthread_barrier_t *barrier) = NULL;
#endif

static int (*pthread_mutex_lock_real)(pthread_mutex_t*) = NULL;
static int (*pthread_mutex_trylock_real)(pthread_mutex_t*) = NULL;
static int (*pthread_mutex_timedlock_real)(pthread_mutex_t*,const struct timespec *) = NULL;
static int (*pthread_mutex_unlock_real)(pthread_mutex_t*) = NULL;

#if 0
/* HSG
   instrumenting these routines makes the included examples deadlock when running... 
*/
static int (*pthread_cond_signal_real)(pthread_cond_t*) = NULL;
static int (*pthread_cond_broadcast_real)(pthread_cond_t*) = NULL;
static int (*pthread_cond_wait_real)(pthread_cond_t*,pthread_mutex_t*) = NULL;
static int (*pthread_cond_timedwait_real)(pthread_cond_t*,pthread_mutex_t*,const struct timespec *) = NULL;
#endif

static int (*pthread_rwlock_rdlock_real)(pthread_rwlock_t *) = NULL;
static int (*pthread_rwlock_tryrdlock_real)(pthread_rwlock_t *) = NULL;
static int (*pthread_rwlock_timedrdlock_real)(pthread_rwlock_t *, const struct timespec *) = NULL;
static int (*pthread_rwlock_wrlock_real)(pthread_rwlock_t *) = NULL;
static int (*pthread_rwlock_trywrlock_real)(pthread_rwlock_t *) = NULL;
static int (*pthread_rwlock_timedwrlock_real)(pthread_rwlock_t *, const struct timespec *) = NULL;
static int (*pthread_rwlock_unlock_real)(pthread_rwlock_t *) = NULL;

static pthread_mutex_t extrae_pthread_create_mutex;

#endif /* PIC */

static void GetpthreadHookPoints (int rank)
{
#if defined(PIC)
	/* Create mutex to protect pthread_create calls */
	pthread_mutex_init (&extrae_pthread_create_mutex, NULL);

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

#if defined(HAVE_PTHREAD_BARRIER_WAIT)
  	/* Obtain @ for pthread_barrier_wait */
	pthread_barrier_wait_real =
		(int(*)(pthread_barrier_t *)) dlsym (RTLD_NEXT, "pthread_barrier_wait");
	if (pthread_barrier_wait_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_barrier_wait in DSOs!!\n");
#endif

	/* Obtain @ for pthread_detach */
	pthread_detach_real = (int(*)(pthread_t)) dlsym (RTLD_NEXT, "pthread_detach");
	if (pthread_detach_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_detach in DSOs!!\n");

	/* Obtain @ for pthread_exit */
	pthread_exit_real = (void(*)(void*)) dlsym (RTLD_NEXT, "pthread_exit");
	if (pthread_exit_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_exit in DSOs!!\n");

	/* Obtain @ for pthread_mutex_lock */
	pthread_mutex_lock_real = (int(*)(pthread_mutex_t*)) dlsym (RTLD_NEXT, "pthread_mutex_lock");
	if (pthread_mutex_lock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_lock in DSOs!!\n");
	
	/* Obtain @ for pthread_mutex_unlock */
	pthread_mutex_unlock_real = (int(*)(pthread_mutex_t*)) dlsym (RTLD_NEXT, "pthread_mutex_unlock");
	if (pthread_mutex_unlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_unlock in DSOs!!\n");
	
	/* Obtain @ for pthread_mutex_trylock */
	pthread_mutex_trylock_real = (int(*)(pthread_mutex_t*)) dlsym (RTLD_NEXT, "pthread_mutex_trylock");
	if (pthread_mutex_trylock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_trylock in DSOs!!\n");

	/* Obtain @ for pthread_mutex_timedlock */
	pthread_mutex_timedlock_real = (int(*)(pthread_mutex_t*,const struct timespec*)) dlsym (RTLD_NEXT, "pthread_mutex_timedlock");
	if (pthread_mutex_timedlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_mutex_timedlock in DSOs!!\n");

#if 0
	/* Obtain @ for pthread_cond_signal */
	pthread_cond_signal_real = (int(*)(pthread_cond_t*)) dlsym (RTLD_NEXT, "pthread_cond_signal");
	if (pthread_cond_signal_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_cond_signal in DSOs!!\n");
	
	/* Obtain @ for pthread_cond_broadcast */
	pthread_cond_broadcast_real = (int(*)(pthread_cond_t*)) dlsym (RTLD_NEXT, "pthread_cond_broadcast");
	if (pthread_cond_broadcast_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_cond_broadcast in DSOs!!\n");
	
	/* Obtain @ for pthread_cond_wait */
	pthread_cond_wait_real = (int(*)(pthread_cond_t*,pthread_mutex_t*)) dlsym (RTLD_NEXT, "pthread_cond_wait");
	if (pthread_cond_wait_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_cond_wait in DSOs!!\n");
	
	/* Obtain @ for pthread_cond_timedwait */
	pthread_cond_timedwait_real = (int(*)(pthread_cond_t*,pthread_mutex_t*,const struct timespec*)) dlsym (RTLD_NEXT, "pthread_cond_timedwait");
	if (pthread_cond_timedwait_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_cond_timedwait in DSOs!!\n");
#endif
	
	/* Obtain @ for pthread_rwlock_rdlock */
	pthread_rwlock_rdlock_real = (int(*)(pthread_rwlock_t*)) dlsym (RTLD_NEXT, "pthread_rwlock_rdlock");
	if (pthread_rwlock_rdlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_rwlock_rdlock in DSOs!!\n");
	
	/* Obtain @ for pthread_rwlock_tryrdlock */
	pthread_rwlock_tryrdlock_real = (int(*)(pthread_rwlock_t*)) dlsym (RTLD_NEXT, "pthread_rwlock_tryrdlock");
	if (pthread_rwlock_tryrdlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_rwlock_tryrdlock in DSOs!!\n");
	
	/* Obtain @ for pthread_rwlock_timedrdlock */
	pthread_rwlock_timedrdlock_real = (int(*)(pthread_rwlock_t *, const struct timespec *)) dlsym (RTLD_NEXT, "pthread_rwlock_timedrdlock");
	if (pthread_rwlock_timedrdlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_rwlock_timedrdlock in DSOs!!\n");
	
	/* Obtain @ for pthread_rwlock_rwlock */
	pthread_rwlock_wrlock_real = (int(*)(pthread_rwlock_t*)) dlsym (RTLD_NEXT, "pthread_rwlock_wrlock");
	if (pthread_rwlock_wrlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_rwlock_wrlock in DSOs!!\n");
	
	/* Obtain @ for pthread_rwlock_tryrwlock */
	pthread_rwlock_trywrlock_real = (int(*)(pthread_rwlock_t*)) dlsym (RTLD_NEXT, "pthread_rwlock_trywrlock");
	if (pthread_rwlock_trywrlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_rwlock_trywrlock in DSOs!!\n");
	
	/* Obtain @ for pthread_rwlock_timedrwlock */
	pthread_rwlock_timedwrlock_real = (int(*)(pthread_rwlock_t *, const struct timespec *)) dlsym (RTLD_NEXT, "pthread_rwlock_timedwrlock");
	if (pthread_rwlock_timedwrlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_rwlock_timedwrlock in DSOs!!\n");

	/* Obtain @ for pthread_rwlock_unlock */
	pthread_rwlock_unlock_real = (int(*)(pthread_rwlock_t*)) dlsym (RTLD_NEXT, "pthread_rwlock_unlock");
	if (pthread_rwlock_unlock_real == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find pthread_rwlock_unlock in DSOs!!\n");
#else
	fprintf (stderr, PACKAGE_NAME": Warning! pthread instrumentation requires linking with shared library!\n");
#endif /* PIC */
}

/*

   INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	 INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE

*/

#if defined(PIC)

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
	void *res = 0;

	Backend_SetpThreadIdentifier (i->pthreadID);

	/* Notify the calling thread */
	pthread_mutex_lock_real (&(i->lock));
	pthread_cond_signal (&(i->wait));
	pthread_mutex_unlock_real (&(i->lock));

	Backend_Enter_Instrumentation (2);
	Probe_pthread_Function_Entry (routine);
	Backend_Leave_Instrumentation ();

	res = routine (arg);

	/* Another thread may have called Extrae_fini() while executing
	   the routine thread */
	if (Extrae_is_initialized_Wrapper() != EXTRAE_NOT_INITIALIZED)
	{
		Probe_pthread_Function_Exit ();
		Backend_Leave_Instrumentation ();
		Backend_Flush_pThread (pthread_self());
	}

	return res;
}

int pthread_create (pthread_t* p1, const pthread_attr_t* p2,
	void *(*p3) (void *), void* p4)
{
	static int pthread_library_depth = 0;
	int res = 0;
	struct pthread_create_info i;

	if (pthread_create_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_create (%p, %p, %p, %p)\n", p1, p2, p3, p4);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_create_real at %p\n", pthread_create_real);
#endif

	if (pthread_create_real != NULL && EXTRAE_INITIALIZED() && Extrae_get_pthread_tracing())
	{
		/* This is a bit tricky.
		   Some OSes (like FreeBSD) delegates the pthread library initialization
		   to the very first call of pthread_create. In order to initialize the 
		   library the OS calls pthread_create again to create the structure for
		   the main thread.
		   So, pthread_library_depth > 0 controls this situation
		*/

		/* Protect creation, just one at a time */
		pthread_mutex_lock_real (&extrae_pthread_create_mutex);

		if (0 == pthread_library_depth)
		{
			pthread_library_depth++;

			Backend_Enter_Instrumentation (1);

			Probe_pthread_Create_Entry (p3);
		
			pthread_cond_init (&(i.wait), NULL);
			pthread_mutex_init (&(i.lock), NULL);
			pthread_mutex_lock_real (&(i.lock));

			i.arg = p4;
			i.routine = p3;
			i.pthreadID = Backend_getNumberOfThreads();

			Backend_ChangeNumberOfThreads (i.pthreadID+1);

			res = pthread_create_real (p1, p2, pthread_create_hook, (void*) &i);

			if (0 == res)
			{
				/* if succeded, wait for a completion on copy the info */
				pthread_cond_wait (&(i.wait), &(i.lock));

				Backend_SetpThreadID (p1, i.pthreadID);
			}

			pthread_mutex_unlock_real (&(i.lock));
			pthread_mutex_destroy (&(i.lock));
			pthread_cond_destroy (&(i.wait));

			Probe_pthread_Create_Exit ();
			Backend_Leave_Instrumentation ();

			pthread_library_depth--;
		}
		else
			res = pthread_create_real (p1, p2, p3, p4);

		/* Stop protecting the region, more pthread creations can enter */
		pthread_mutex_unlock_real (&extrae_pthread_create_mutex);
	}
	else if (pthread_create_real != NULL)
	{
		res = pthread_create_real (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_create was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_join (pthread_t p1, void **p2)
{
	int res = 0;

	if (pthread_join_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_join (%p, %p)\n", p1, p2);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_join_real at %p\n", pthread_join_real);
#endif

	if (pthread_join_real != NULL && EXTRAE_INITIALIZED() && Extrae_get_pthread_tracing())
	{
		Backend_Enter_Instrumentation (2);
		Probe_pthread_Join_Entry ();

		res = pthread_join_real (p1, p2);

		/* Careful, initialized state may have changed after join! */
		if (Extrae_is_initialized_Wrapper())
		{
			Backend_Flush_pThread (p1);

			Probe_pthread_Join_Exit ();
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_join_real != NULL)
	{
		res = pthread_join_real (p1, p2);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_join was not hooked\n");
		exit (-1);
	}
	return res;
}

void pthread_exit (void *p1)
{
	if (pthread_exit_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_exit (%p)\n", p1);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_exit_real at %p\n", pthread_exit_real);
#endif

	if (pthread_exit_real != NULL && EXTRAE_INITIALIZED() && Extrae_get_pthread_tracing())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (2);
			Probe_pthread_Function_Exit();
			Probe_pthread_Exit_Entry();
			Backend_Leave_Instrumentation ();
			Backend_Flush_pThread (pthread_self());
		}

		pthread_exit_real (p1);
	}
	else if (pthread_exit_real != NULL)
	{
		pthread_exit_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_exit was not hooked\n");
		exit (-1);
	}
}

int pthread_detach (pthread_t p1)
{
	int res = 0;

	if (pthread_detach_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_detach (%p)\n", p1);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_detach_real at %p\n", pthread_detach_real);
#endif

	if (pthread_detach_real != NULL && EXTRAE_INITIALIZED() && Extrae_get_pthread_tracing())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_Detach_Entry ();
			res = pthread_detach_real (p1);
			Probe_pthread_Detach_Exit ();
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_detach_real != NULL)
	{
		res = pthread_detach_real (p1);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_detach was not hooked\n");
		exit (-1);
	}
	return res;
}

int pthread_mutex_lock (pthread_mutex_t *m)
{
	int res = 0;

	if (pthread_mutex_lock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_lock (%p)\n", m);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_lock_real at %p\n", pthread_mutex_lock_real);
#endif

	/* Caution! pthread_exit() seems to call pthread_mutex_lock */
	if (pthread_mutex_lock_real != NULL && EXTRAE_INITIALIZED()
	  && Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_mutex_lock_Entry (m);
			res = pthread_mutex_lock_real (m);
			Probe_pthread_mutex_lock_Exit (m);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_mutex_lock_real != NULL)
	{
		res = pthread_mutex_lock_real (m);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_mutex_lock was not hooked\n");
		exit (-1);
	}
	return res;
}

int pthread_mutex_trylock (pthread_mutex_t *m)
{
	int res = 0;

	if (pthread_mutex_trylock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_trylock (%p)\n", m);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_trylock_real at %p\n", pthread_mutex_trylock_real);
#endif

	if (pthread_mutex_trylock_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_mutex_lock_Entry (m);
			res = pthread_mutex_trylock_real (m);
			Probe_pthread_mutex_lock_Exit (m);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_mutex_trylock_real != NULL)
	{
		res = pthread_mutex_trylock_real (m);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_mutex_trylock was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_mutex_timedlock(pthread_mutex_t *m, const struct timespec *t)
{
	int res = 0;

	if (pthread_mutex_timedlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_timedlock (%p,%p)\n", m,t);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_timedlock_real at %p\n", pthread_mutex_timedlock_real);
#endif

	if (pthread_mutex_timedlock_real != NULL && EXTRAE_INITIALIZED() && 
	 Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_mutex_lock_Entry (m);
			res = pthread_mutex_timedlock_real (m, t);
			Probe_pthread_mutex_lock_Exit (m);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_mutex_timedlock_real != NULL)
	{
		res = pthread_mutex_timedlock_real (m, t);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_mutex_timedlock was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_mutex_unlock (pthread_mutex_t *m)
{
	int res = 0;

	if (pthread_mutex_unlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_unlock (%p)\n", m);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_mutex_unlock_real at %p\n", pthread_mutex_unlock_real);
#endif

	/* Caution! pthread_exit() seems to call pthread_mutex_lock */
	if (pthread_mutex_unlock_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_mutex_unlock_Entry (m);
			res = pthread_mutex_unlock_real (m);
			Probe_pthread_mutex_unlock_Exit (m);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_mutex_unlock_real != NULL)
	{
		res = pthread_mutex_unlock_real (m);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_mutex_unlock was not hooked\n");
		exit (-1);
	}

	return res;
}


#if 0
int pthread_cond_signal (pthread_cond_t *c)
{
	int res = 0;

	if (pthread_cond_signal_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_signal (%p)\n", c);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_signal_real at %p\n", pthread_cond_signal_real);
#endif

	if (pthread_cond_signal_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_cond_signal_Entry (c);
			res = pthread_cond_signal_real (c);
			Probe_pthread_cond_signal_Exit (c);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_cond_signal_real != NULL)
	{
		res = pthread_cond_signal_real (c);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_cond_signal was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_cond_broadcast (pthread_cond_t *c)
{
	int res = 0;

	if (pthread_cond_broadcast_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_broadcast (%p)\n", c);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_broadcast_real at %p\n", pthread_cond_broadcast_real);
#endif

	if (pthread_cond_broadcast_real != NULL && EXTRAE_INITIALIZED() &&
	   Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_cond_broadcast_Entry (c);
			res = pthread_cond_broadcast_real (c);
			Probe_pthread_cond_broadcast_Exit (c);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_cond_broadcast_real != NULL)
	{
		res = pthread_cond_broadcast_real (c);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_cond_broadcast was not hooked\n");
		exit (-1);
	}
	return res;
}

int pthread_cond_wait (pthread_cond_t *c, pthread_mutex_t *m)
{
	int res = 0;

	if (pthread_cond_wait_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_wait (%p, %p)\n", c, m);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_wait_real at %p\n", pthread_cond_wait_real);
#endif

	if (pthread_cond_wait_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_cond_wait_Entry (c);
			res = pthread_cond_wait_real (c, m);
			Probe_pthread_cond_wait_Exit (c);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_cond_wait_real != NULL)
	{
		res = pthread_cond_wait_real (c, m);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_cond_wait was not hooked\n");
		exit (-1);
	}
	return res;
}

int pthread_cond_timedwait (pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *t)
{
	int res = 0;

	if (pthread_cond_timedwait_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_timedwait (%p,%p,%p)\n", c, m, t);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_cond_wait_real at %p\n", pthread_cond_wait_real);
#endif

	if (pthread_cond_timedwait_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_cond_wait_Entry (c);
			res = pthread_cond_timedwait_real (c,m,t);
			Probe_pthread_cond_wait_Exit (c);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_cond_timedwait_real != NULL)
	{
		res = pthread_cond_timedwait_real (c,m,t);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_cond_timedwait was not hooked\n");
		exit (-1);
	}

	return res;
}
#endif

int pthread_rwlock_rdlock (pthread_rwlock_t *l)
{
	int res = 0;

	if (pthread_rwlock_rdlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_rdlock (%p)\n", l);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_rdlock_real at %p\n", pthread_rwlock_rdlock_real);
#endif

	if (pthread_rwlock_rdlock_real != NULL && EXTRAE_INITIALIZED() &&
	 Extrae_get_pthread_tracing() &&  Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_rwlock_lockrd_Entry (l);
			res = pthread_rwlock_rdlock_real (l);
			Probe_pthread_rwlock_lockrd_Exit (l);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_rwlock_rdlock_real != NULL)
	{
		res = pthread_rwlock_rdlock_real (l);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_rwlock_rdlock was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_rwlock_tryrdlock(pthread_rwlock_t *l)
{
	int res = 0;

	if (pthread_rwlock_tryrdlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_tryrdlock (%p)\n", l);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_tryrdlock_real at %p\n", pthread_rwlock_tryrdlock_real);
#endif

	if (pthread_rwlock_tryrdlock_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_rwlock_lockrd_Entry (l);
			res = pthread_rwlock_tryrdlock_real (l);
			Probe_pthread_rwlock_lockrd_Exit (l);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_rwlock_tryrdlock_real != NULL)
	{
		res = pthread_rwlock_tryrdlock_real (l);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_rwlock_tryrdlock was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_rwlock_timedrdlock(pthread_rwlock_t *l, const struct timespec *t)
{
	int res = 0;

	if (pthread_rwlock_timedrdlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_timedrdlock (%p, %p)\n", l, t);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_timedrdlock_real at %p\n", pthread_rwlock_timedrdlock_real);
#endif

	if (pthread_rwlock_timedrdlock_real != NULL && EXTRAE_INITIALIZED() &&
	 Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_rwlock_lockrd_Entry (l);
			res = pthread_rwlock_timedrdlock_real (l, t);
			Probe_pthread_rwlock_lockrd_Exit (l);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_rwlock_timedrdlock_real != NULL)
	{
		res = pthread_rwlock_timedrdlock_real (l, t);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_rwlock_timedrdlock was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_rwlock_wrlock(pthread_rwlock_t *l)
{
	int res = 0;

	if (pthread_rwlock_wrlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_wrlock (%p)\n", l);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_wrlock_real at %p\n", pthread_rwlock_wrlock_real);
#endif

	if (pthread_rwlock_wrlock_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_rwlock_lockwr_Entry (l);
			res = pthread_rwlock_wrlock_real (l);
			Probe_pthread_rwlock_lockwr_Exit (l);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_rwlock_wrlock_real != NULL)
	{
		res = pthread_rwlock_wrlock_real (l);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_rwlock_wrlock was not hooked\n");
		exit (-1);
	}
	return res;
}

int pthread_rwlock_trywrlock(pthread_rwlock_t *l)
{
	int res = 0;

	if (pthread_rwlock_trywrlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_trywrlock (%p)\n", l);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_trywrlock_real at %p\n", pthread_rwlock_trywrlock_real);
#endif

	if (pthread_rwlock_trywrlock_real != NULL && EXTRAE_INITIALIZED() &&
	 Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_rwlock_lockwr_Entry (l);
			res = pthread_rwlock_trywrlock_real (l);
			Probe_pthread_rwlock_lockwr_Exit (l);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_rwlock_trywrlock_real != NULL)
	{
		res = pthread_rwlock_trywrlock_real (l);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_rwlock_trywrlock was not hooked\n");
		exit (-1);
	}
	return res;
}

int pthread_rwlock_timedwrlock(pthread_rwlock_t *l, const struct timespec *t)
{
	int res = 0;

	if (pthread_rwlock_timedwrlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_timedwrlock (%p)\n", l);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_timedwrlock_real at %p\n", pthread_rwlock_timedwrlock_real);
#endif

	if (pthread_rwlock_timedwrlock_real != NULL && EXTRAE_INITIALIZED() &&
	 Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_rwlock_lockwr_Entry (l);
			res = pthread_rwlock_timedwrlock_real (l, t);
			Probe_pthread_rwlock_lockwr_Exit (l);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_rwlock_timedwrlock_real != NULL)
	{
		res = pthread_rwlock_timedwrlock_real (l, t);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": pthread_rwlock_timedwrlock was not hooked\n");
		exit (-1);
	}

	return res;
}

int pthread_rwlock_unlock(pthread_rwlock_t *l)
{
	int res = 0;

	if (pthread_rwlock_unlock_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG) && defined(DEBUG_MUTEX)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_unlock (%p)\n", l);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_rwlock_unlock_real at %p\n", pthread_rwlock_unlock_real);
#endif

	if (pthread_rwlock_unlock_real != NULL && EXTRAE_INITIALIZED() &&
	 Extrae_get_pthread_tracing() && Extrae_get_pthread_instrument_locks())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (1);
			Probe_pthread_rwlock_unlock_Entry (l);
			res = pthread_rwlock_unlock_real (l);
			Probe_pthread_rwlock_unlock_Exit (l);
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_rwlock_unlock_real != NULL)
	{
		res = pthread_rwlock_unlock_real (l);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": pthread_rwlock_unlock was not hooked\n");
		exit (-1);
	}

	return res;
}

#if defined(HAVE_PTHREAD_BARRIER_WAIT)
int pthread_barrier_wait (pthread_barrier_t *barrier)
{
	int res = 0;

	if (pthread_barrier_wait_real == NULL)
		GetpthreadHookPoints(0);

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_barrier_wait (%p)\n", barrier);
	fprintf (stderr, PACKAGE_NAME": DEBUG: pthread_barrier_wait_real at %p\n", pthread_barrier_wait_real);
#endif
	if (pthread_barrier_wait_real != NULL && EXTRAE_INITIALIZED() &&
	  Extrae_get_pthread_tracing())
	{
		if (!Backend_ispThreadFinished(THREADID))
		{
			Backend_Enter_Instrumentation (2);
			Probe_pthread_Barrier_Wait_Entry ();
			res = pthread_barrier_wait_real (barrier);
			Probe_pthread_Barrier_Wait_Exit ();
			Backend_Leave_Instrumentation ();
		}
	}
	else if (pthread_barrier_wait_real != NULL)
	{
		res = pthread_barrier_wait_real (barrier);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error pthread_barrier_wait was not hooked\n");
		exit (-1);
	}
	return res;
}
#endif /* OS_ANDROID */
#endif /* PIC */

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


