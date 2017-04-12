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

#ifdef HAVE_ASSERT_H
# include <assert.h>
#endif
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_ASSERT_H
# include <assert.h>
#endif
#ifdef HAVE_DFLCN_H
# define __USE_GNU
#  include <dlfcn.h>
# undef __USE_GNU
#endif

#include "misc_interface.h"
#include "misc_wrapper.h"
#include "wrapper.h"
#include "threadid.h"

#include "omp-probe.h"
#include "omp-events.h"
#include "ompt-helper.h"
#include "ompt-target.h"


// # define EMPTY_OMPT_CALLBACKS /* For Benchmarking purposes */
// #define DEBUG
// #define DEBUG_THREAD

//*****************************************************************************
// interface operations
//*****************************************************************************

int ompt_enabled = FALSE;

int (*ompt_set_callback_fn)(ompt_event_t, ompt_callback_t) = NULL;
ompt_thread_id_t (*ompt_get_thread_id_fn)(void) = NULL;

typedef struct omptthid_threadid_st
{
	ompt_thread_id_t thid;
	unsigned threadid;
	int in_use;
} omptthid_threadid_t;

static omptthid_threadid_t *ompt_thids = NULL;
static unsigned n_ompt_thids = 0;
static pthread_mutex_t mutex_thids = PTHREAD_MUTEX_INITIALIZER;

static int ompt_targets_initialized = 0;

/* Extrae_OMPT_register_ompt_thread_id
   registers a OMPT thread ompt_thread_id_t and assign it a Paraver thread with
   an identifier threadid */
static void Extrae_OMPT_register_ompt_thread_id (ompt_thread_id_t ompt_thid,
	unsigned threadid)
{
	int found_empty = FALSE;
	unsigned u;
	unsigned free_slot = 0;

	pthread_mutex_lock (&mutex_thids);

	/* Search for an empty slot, if any */
	for (u = 0; u < n_ompt_thids; u++)
		if (!ompt_thids[u].in_use)
		{
			found_empty = TRUE;
			free_slot = u;
			break;
		}

#if defined(DEBUG)
	printf ("REGISTER_THREAD[] => { found_empty = %d, free_slot = %u }\n", found_empty, free_slot);
#endif

	/* If not empty, allocate space for a new entry */
	if (!found_empty)
	{
		ompt_thids = (omptthid_threadid_t*) realloc (ompt_thids,
		  (n_ompt_thids+1)*sizeof(omptthid_threadid_t));
		assert (ompt_thids != NULL);
		free_slot = n_ompt_thids;
		n_ompt_thids++;
	}

#if defined(DEBUG)
	if (!found_empty)
		printf ("REGISTERING-new(slot=%u) { ompt_thid=%lu, threadid=%u }\n", free_slot, ompt_thid, threadid);
	else
		printf ("REGISTERING-reused(slot=%u) { ompt_thid=%lu, threadid=%u }\n", free_slot, ompt_thid, threadid);
#endif

	/* Set slot info on 'free_slot' */
	ompt_thids[free_slot].thid     = ompt_thid;
	ompt_thids[free_slot].threadid = threadid;
	ompt_thids[free_slot].in_use   = TRUE;

	pthread_mutex_unlock (&mutex_thids);
}

/* Extrae_OMPT_unregister_ompt_thread_id
   unregisters the OMPT thread with identifier ompt_thid
*/
static void Extrae_OMPT_unregister_ompt_thread_id (ompt_thread_id_t ompt_thid)
{
	unsigned u;

	pthread_mutex_lock (&mutex_thids);

#if defined(DEBUG)
	printf ("UNREGISTERING(ompt_thid %lu)\n", ompt_thid);
#endif

	for (u = 0; u < n_ompt_thids; u++)
		if (ompt_thids[u].thid == ompt_thid && ompt_thids[u].in_use)
		{
			ompt_thids[u].in_use = FALSE;
			break;
		}

	pthread_mutex_unlock (&mutex_thids);
}

/* Extrae_OMPT_threadid
   queries to OMPT for the OMPT thread id and then returns for the Paraver
   registered thread identifier.  */
static unsigned Extrae_OMPT_threadid (void)
{
	ompt_thread_id_t thd = ompt_get_thread_id_fn();
	unsigned u;

#if defined(NEED_MUTEX_TO_GET_THREADID)
	pthread_mutex_lock (&mutex_thids);
#endif

	/* If we haven't tracked any thread atm, return thid 0 */
	if (n_ompt_thids == 0)
	{
#if defined(NEED_MUTEX_TO_GET_THREADID)
		pthread_mutex_unlock (&mutex_thids);
#endif
		return 0;
	}

	for (u = 0; u < n_ompt_thids; u++)
	{
#if defined(DEBUG_THREAD)
		printf ("SEARCHING (thread=%lu) [slot=%u/%u] {ompt_thid=%lu,in_use=%d,threadid=%u}\n",
		 thd, u, n_ompt_thids, ompt_thids[u].thid, ompt_thids[u].in_use, ompt_thids[u].threadid);
#endif
		if (ompt_thids[u].thid == thd && ompt_thids[u].in_use)
		{
#if defined(NEED_MUTEX_TO_GET_THREADID)
			pthread_mutex_unlock (&mutex_thids);
#endif
#if defined(DEBUG_THREAD)
			printf ("RETURNING %u\n", ompt_thids[u].threadid);
#endif
			return ompt_thids[u].threadid;
		}
	}

#if defined(NEED_MUTEX_TO_GET_THREADID)
	pthread_mutex_unlock (&mutex_thids);
#endif
	fprintf (stderr, "OMPTOOL: Failed to search OpenMP(T) thread %lu\n", thd);
	fprintf (stderr, "OMPTOOL: Registered threads are (%u): ", n_ompt_thids);
	for (u = 0; u < n_ompt_thids; u++)
		fprintf (stderr, "ompt thid %lu valid %d ", ompt_thids[u].thid, ompt_thids[u].in_use);
	fprintf (stderr, "\n");

	assert (1 != 1);
	return 0;
}


#if defined(DEBUG)
# define PROTOTYPE_MESSAGE_NOTHREAD(fmt, ...) \
   printf ("THREAD=XX/XX %s" fmt "\n", \
    __func__, \
    ## __VA_ARGS__)
# if defined(__INTEL_COMPILER)
#  define PROTOTYPE_MESSAGE(fmt, ...) \
    printf ("THREAD=%u/%lu %s" fmt "\n", \
     0u, \
     0ull, \
     __func__, \
     ## __VA_ARGS__)
# else
#  define PROTOTYPE_MESSAGE(fmt, ...) \
    printf ("THREAD=%u/%lu %s" fmt "\n", \
     Extrae_OMPT_threadid(), \
     ompt_get_thread_id_fn(), \
     __func__, \
     ## __VA_ARGS__)
# endif
#else
# define PROTOTYPE_MESSAGE(...)
# define PROTOTYPE_MESSAGE_NOTHREAD(...)
#endif

static pthread_mutex_t mutex_init_threads = PTHREAD_MUTEX_INITIALIZER;

void Extrae_OMPT_event_thread_begin (ompt_thread_type_t type,
	ompt_thread_id_t thid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	pthread_mutex_lock (&mutex_init_threads);

	/* Get last thread id in instrumentation rte */
	unsigned threads = Backend_getNumberOfThreads();

	PROTOTYPE_MESSAGE_NOTHREAD(" TYPE %d (worker == %d) THID %lu (threads=%u)", type, ompt_thread_worker, thid, threads);

	if (type == ompt_thread_initial)
	{
		/* If I'm the initial thread, I'm thread 0 on the Paraver trace */
		Extrae_OMPT_register_ompt_thread_id (thid, 0);
	}
	else
	{
		/* If I'm not the initial thread, then look for how many threads are
		   there and then give it the next id */
		Extrae_OMPT_register_ompt_thread_id (thid, threads);
		Backend_ChangeNumberOfThreads (threads + 1);
	}

	pthread_mutex_unlock (&mutex_init_threads);
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_thread_end (ompt_thread_type_t type,
	ompt_thread_id_t thid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	pthread_mutex_lock (&mutex_init_threads);

	/* Get last thread id in instrumentation rte */
	unsigned threads = Backend_getNumberOfThreads();

	PROTOTYPE_MESSAGE(" TYPE %d THID %ld", type, thid);

	if (type == ompt_thread_worker)
	{
		Extrae_OMPT_unregister_ompt_thread_id (thid);
		Backend_ChangeNumberOfThreads (threads-1);
	}

	pthread_mutex_unlock (&mutex_init_threads);
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_loop_begin (ompt_parallel_id_t pid, ompt_task_id_t tid,
	const void *wf)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Loop_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_loop_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Loop_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_parallel_begin (ompt_task_id_t tid, ompt_frame_t *ptf,
	ompt_parallel_id_t pid, uint32_t req_team_size, const void *pf,
	ompt_invoker_t invoker)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(ptf);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(req_team_size);
	UNREFERENCED_PARAMETER(invoker);

	PROTOTYPE_MESSAGE(" (%ld, %p, %ld, %u, %p)", tid, ptf, pid, req_team_size, pf);

	/* At this point we need to register the parallel region with identifier (pid)
	   to the outlined code (pointed by pf). This is necessary because there won't
	   be any further reference to pf but to pid */
	Extrae_OMPT_register_ompt_parallel_id_pf (pid, pf);
	Extrae_OpenMP_ParRegion_Entry ();
	Extrae_OpenMP_EmitTaskStatistics();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_parallel_end (ompt_parallel_id_t pid, ompt_task_id_t tid,
	ompt_invoker_t invoker)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(invoker);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);

	/* At this point we can unregister the association between pid - pf */
	Extrae_OMPT_unregister_ompt_parallel_id_pf (pid);
	Extrae_OpenMP_ParRegion_Exit();
	Extrae_OpenMP_EmitTaskStatistics();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_barrier_begin (ompt_parallel_id_t pid, ompt_task_id_t tid,
	const void *codeptr_ra)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(codeptr_ra);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Barrier_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_barrier_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Barrier_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_barrier_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_barrier_end (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_master_begin(ompt_parallel_id_t pid, ompt_task_id_t tid,
	const void *codeptr_ra)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(codeptr_ra);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Master_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_master_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Master_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_sections_begin (ompt_parallel_id_t pid, ompt_task_id_t tid,
	const void *codeptr_ra)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(codeptr_ra);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, codeptr_ra);
	Extrae_OMPT_Sections_Entry();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_sections_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Sections_Exit();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_single_others_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE, nothing atm?
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_single_others_end (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE, nothing atm?
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_single_in_block_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid, const void *codeptr_ra)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(codeptr_ra);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, codeptr_ra);
	Extrae_OMPT_Single_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_single_in_block_end (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Single_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_task_begin (ompt_task_id_t ptid, const ompt_frame_t *ptf,
	ompt_task_id_t tid, const void *ntf)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(ptid);
	UNREFERENCED_PARAMETER(ptf);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %p, %ld, %p)", ptid, ptf, tid, ntf);

	/* Similar to Extrae_OMPT_event_parallel_begin, ntf won't be available
	   whenever additional activity to task tid occurs. We need to store the
	   relation between tid and its code (ntf). In this case, the task is NOT
	   implicit. */
	Extrae_OMPT_register_ompt_task_id_tf (tid, ntf, FALSE);
	Extrae_OpenMP_Notify_NewInstantiatedTask();
	//Extrae_OpenMP_TaskUF_Entry (ntf); NOTE: Task does not start running here (HSG, for IBM)!
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_task_end (ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	PROTOTYPE_MESSAGE(" (%ld)", tid);
	Extrae_OpenMP_Notify_NewExecutedTask();
	if (Extrae_OMPT_tf_task_id_is_running(tid))
		// If this task was not marked as running at switch, mark it here
		Extrae_OMPT_OpenMP_TaskUF_Exit (tid);
	Extrae_OMPT_unregister_ompt_task_id_tf (tid);
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_taskwait_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_taskwait_end (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_taskwait_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid, const void *codeptr_ra)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(codeptr_ra);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Taskwait_Entry ();
	Extrae_OpenMP_EmitTaskStatistics();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_taskwait_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Taskwait_Exit ();
	Extrae_OpenMP_EmitTaskStatistics();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_taskgroup_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Taskgroup_Entry();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_taskgroup_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Taskgroup_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_taskgroup_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_taskgroup_end (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_workshare_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid, const void *codeptr_ra)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(codeptr_ra);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, codeptr_ra);
	Extrae_OMPT_Workshare_Entry();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_workshare_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Workshare_Exit();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_idle_begin (ompt_thread_id_t thid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(thid);

	PROTOTYPE_MESSAGE(" (THID = %ld)", thid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_idle_end (ompt_thread_id_t thid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(thid);

	PROTOTYPE_MESSAGE(" (THID = %ld)", thid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_release_lock (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Unlock_Exit();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_release_nest_lock_last (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Unlock_Exit();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_release_critical (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Unlock_Exit();
	Extrae_OMPT_Critical_Exit();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_release_ordered (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Unlock_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_release_atomic (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Unlock_Exit ();
	Extrae_OMPT_Atomic_Exit();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_implicit_task_begin (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	/* Similar to Extrae_OMPT_event_task_begin but in this case two things differ:
	   1) we don't have the code associated to the function, so it is derived from the
	   code associated to the parallel function. 2) We need to mark this task as
	   implicit task */
	Extrae_OMPT_register_ompt_task_id_tf (tid, Extrae_OMPT_get_pf_parallel_id(pid), TRUE);
	Extrae_OpenMP_UF_Entry (Extrae_OMPT_get_pf_parallel_id(pid));
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_implicit_task_end (ompt_parallel_id_t pid,
	ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_unregister_ompt_task_id_tf (tid);
	Extrae_OpenMP_UF_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_initial_task_begin (ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld)", tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_initial_task_end (ompt_task_id_t tid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld)", tid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_task_switch (ompt_task_id_t stid, ompt_task_id_t rtid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	const void *tf;
	long long taskcounter;
	int implicit;

	/* This event denotes that the previous task was suspended and another resumed */
	PROTOTYPE_MESSAGE(" (%ld, %ld)", stid, rtid);

	/* Leave a task function if it's not the implicit task. The implicit
	   task is automatically instrumented elsewhere */
	if (stid > 0)
		if ((tf = Extrae_OMPT_get_tf_task_id(stid, &implicit, NULL)))
			if (!implicit)
			{
				Extrae_OMPT_OpenMP_TaskUF_Exit (stid);
				Extrae_OMPT_tf_task_id_set_running (stid, FALSE);
			}

	/* Enter a task function if it's not the implicit task. The implicit
	   task is automatically instrumented elsewhere */
	if ((tf = Extrae_OMPT_get_tf_task_id (rtid, &implicit, &taskcounter)))
		if (!implicit)
		{
			Extrae_OMPT_OpenMP_TaskUF_Entry ((UINT64)tf, rtid);
			Extrae_OMPT_tf_task_id_set_running (rtid, TRUE);
			Extrae_OpenMP_TaskID (taskcounter);
		}
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_lock (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_acquired_lock (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Exit ((void*)wid);
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_nest_lock (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_acquired_nest_lock_first (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Exit ((void*)wid);
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_release_nest_lock_prev (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_acquired_nest_lock_next (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	// TODO: EXTRAE
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_critical (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_acquired_critical (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Exit ();
	Extrae_OMPT_Critical_Entry();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_ordered (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_acquired_ordered (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Exit ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_wait_atomic (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Entry ();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_acquired_atomic (ompt_wait_id_t wid)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Exit ();
	Extrae_OMPT_Atomic_Entry();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_control (uint64_t command, uint64_t modifier)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(modifier);

	PROTOTYPE_MESSAGE(" (cmd = %lu, mod = %lu)", command, modifier);

	if (command == 1) /* API spec: start or restart monitoring */
		Extrae_restart_Wrapper();
	else if (command == 2) /* API spec: pause monitoring */
		Extrae_shutdown_Wrapper();
	else if (command == 3) /* API spec: flush tool buffer & continue */
		Extrae_flush_manual_Wrapper();
	else if (command == 4) /* API spec: shutdown */
		Extrae_fini_Wrapper();
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_task_dependence_pair ( /* for new dependence */
	ompt_task_id_t pred_task_id, /* ID of predecessor task */
	ompt_task_id_t succ_task_id  /* ID of successor task */
)
{
#ifndef EMPTY_OMPT_CALLBACKS
	PROTOTYPE_MESSAGE(" (pred_task_id = %lx, succ_task_id = %lx)", pred_task_id,
		succ_task_id);

	Extrae_OMPT_dependence (pred_task_id, succ_task_id);
#endif /* EMPTY_OMPT_CALLBACKS */
}

void Extrae_OMPT_event_task_dependences ( /* for new dependences */
	ompt_task_id_t task_id,        /* ID of task */
	ompt_task_dependence_t *deps,  /* vector of task dependences */
	int ndeps
)
{
#ifndef EMPTY_OMPT_CALLBACKS
	UNREFERENCED_PARAMETER(task_id);
	UNREFERENCED_PARAMETER(deps);
	UNREFERENCED_PARAMETER(ndeps);

	PROTOTYPE_MESSAGE(" (task_id = %d, deps = %p, ndeps = %d)", task_id, deps, ndeps);
#if defined(DEBUG)
	int i;

	for (i = 0; i < ndeps; i++)
		printf ("dependence [%d/%d] : variable @ %p, len = %u, flags = %d\n", i+1, ndeps, deps[i].variable_addr);
#endif

#endif /* EMPTY_OMPT_CALLBACKS */
}

//*****************************************************************************
// interface operations
//*****************************************************************************

struct OMPT_callbacks_st
{
	char* evt_name;
	ompt_event_t evt;
	ompt_callback_t cbk;
};

#define CALLBACK_ENTRY(x,y) { #x, x, (ompt_callback_t) y }

static struct OMPT_callbacks_st ompt_callbacks[] =
{
	CALLBACK_ENTRY (ompt_event_loop_begin, Extrae_OMPT_event_loop_begin),
	CALLBACK_ENTRY (ompt_event_loop_end, Extrae_OMPT_event_loop_end),
	CALLBACK_ENTRY (ompt_event_parallel_begin, Extrae_OMPT_event_parallel_begin),
	CALLBACK_ENTRY (ompt_event_parallel_end, Extrae_OMPT_event_parallel_end),
	CALLBACK_ENTRY (ompt_event_barrier_begin, Extrae_OMPT_event_barrier_begin),
	CALLBACK_ENTRY (ompt_event_barrier_end, Extrae_OMPT_event_barrier_end),
	CALLBACK_ENTRY (ompt_event_wait_barrier_begin, Extrae_OMPT_event_wait_barrier_begin),
	CALLBACK_ENTRY (ompt_event_wait_barrier_end, Extrae_OMPT_event_wait_barrier_end),
	CALLBACK_ENTRY (ompt_event_sections_begin, Extrae_OMPT_event_sections_begin),
	CALLBACK_ENTRY (ompt_event_sections_end, Extrae_OMPT_event_sections_end),
	CALLBACK_ENTRY (ompt_event_task_begin, Extrae_OMPT_event_task_begin),
	CALLBACK_ENTRY (ompt_event_task_end, Extrae_OMPT_event_task_end),
	// CALLBACK_ENTRY (ompt_event_wait_taskwait_begin, Extrae_OMPT_event_wait_taskwait_begin),
	// CALLBACK_ENTRY (ompt_event_wait_taskwait_end, Extrae_OMPT_event_wait_taskwait_end),
	CALLBACK_ENTRY (ompt_event_taskwait_begin, Extrae_OMPT_event_taskwait_begin),
	CALLBACK_ENTRY (ompt_event_taskwait_end, Extrae_OMPT_event_taskwait_end),
	CALLBACK_ENTRY (ompt_event_wait_taskgroup_begin, Extrae_OMPT_event_wait_taskgroup_begin),
	CALLBACK_ENTRY (ompt_event_wait_taskgroup_end, Extrae_OMPT_event_wait_taskgroup_end),
	CALLBACK_ENTRY (ompt_event_taskgroup_begin, Extrae_OMPT_event_taskgroup_begin),
	CALLBACK_ENTRY (ompt_event_taskgroup_end, Extrae_OMPT_event_taskgroup_end),
	CALLBACK_ENTRY (ompt_event_workshare_begin, Extrae_OMPT_event_workshare_begin),
	CALLBACK_ENTRY (ompt_event_workshare_end, Extrae_OMPT_event_workshare_end),
	CALLBACK_ENTRY (ompt_event_idle_begin, Extrae_OMPT_event_idle_begin),
	CALLBACK_ENTRY (ompt_event_idle_end, Extrae_OMPT_event_idle_end),
	CALLBACK_ENTRY (ompt_event_implicit_task_begin, Extrae_OMPT_event_implicit_task_begin),
	CALLBACK_ENTRY (ompt_event_implicit_task_end, Extrae_OMPT_event_implicit_task_end),
	CALLBACK_ENTRY (ompt_event_initial_task_begin, Extrae_OMPT_event_initial_task_begin),
	CALLBACK_ENTRY (ompt_event_initial_task_end, Extrae_OMPT_event_initial_task_end),
	CALLBACK_ENTRY (ompt_event_task_switch, Extrae_OMPT_event_task_switch),
	CALLBACK_ENTRY (ompt_event_wait_lock, Extrae_OMPT_event_wait_lock),
	CALLBACK_ENTRY (ompt_event_thread_begin, Extrae_OMPT_event_thread_begin),
	CALLBACK_ENTRY (ompt_event_thread_end, Extrae_OMPT_event_thread_end),
	CALLBACK_ENTRY (ompt_event_control, Extrae_OMPT_event_control),
	CALLBACK_ENTRY (ompt_event_task_dependences, Extrae_OMPT_event_task_dependences),
	CALLBACK_ENTRY (ompt_event_task_dependence_pair, Extrae_OMPT_event_task_dependence_pair),
 	{ "empty,", (ompt_event_t) 0, 0 },
 };
 
struct OMPT_callbacks_st ompt_callbacks_locks[] =
{
	CALLBACK_ENTRY (ompt_event_master_begin, Extrae_OMPT_event_master_begin),
	CALLBACK_ENTRY (ompt_event_master_end, Extrae_OMPT_event_master_end),
	CALLBACK_ENTRY (ompt_event_single_others_begin, Extrae_OMPT_event_single_others_begin),
	CALLBACK_ENTRY (ompt_event_single_others_end, Extrae_OMPT_event_single_others_end),
	CALLBACK_ENTRY (ompt_event_single_in_block_begin, Extrae_OMPT_event_single_in_block_begin),
	CALLBACK_ENTRY (ompt_event_single_in_block_end, Extrae_OMPT_event_single_in_block_end),
	CALLBACK_ENTRY (ompt_event_release_lock, Extrae_OMPT_event_release_lock),
	CALLBACK_ENTRY (ompt_event_release_nest_lock_last, Extrae_OMPT_event_release_nest_lock_last),
	CALLBACK_ENTRY (ompt_event_release_critical, Extrae_OMPT_event_release_critical),
	CALLBACK_ENTRY (ompt_event_release_ordered, Extrae_OMPT_event_release_ordered),
	CALLBACK_ENTRY (ompt_event_release_atomic, Extrae_OMPT_event_release_atomic),
	CALLBACK_ENTRY (ompt_event_acquired_lock, Extrae_OMPT_event_acquired_lock),
	CALLBACK_ENTRY (ompt_event_wait_nest_lock, Extrae_OMPT_event_wait_nest_lock),
	CALLBACK_ENTRY (ompt_event_acquired_nest_lock_first, Extrae_OMPT_event_acquired_nest_lock_first),
	CALLBACK_ENTRY (ompt_event_release_nest_lock_prev, Extrae_OMPT_event_release_nest_lock_prev),
	CALLBACK_ENTRY (ompt_event_acquired_nest_lock_next, Extrae_OMPT_event_acquired_nest_lock_next),
	CALLBACK_ENTRY (ompt_event_wait_critical, Extrae_OMPT_event_wait_critical),
	CALLBACK_ENTRY (ompt_event_acquired_critical, Extrae_OMPT_event_acquired_critical),
	CALLBACK_ENTRY (ompt_event_wait_ordered, Extrae_OMPT_event_wait_ordered),
	CALLBACK_ENTRY (ompt_event_acquired_ordered, Extrae_OMPT_event_acquired_ordered),
	CALLBACK_ENTRY (ompt_event_wait_atomic, Extrae_OMPT_event_wait_atomic),
	CALLBACK_ENTRY (ompt_event_acquired_atomic, Extrae_OMPT_event_acquired_atomic),
	{ "empty,", (ompt_event_t) 0, 0 },
};
 
typedef enum {
	OMPT_RTE_IBM,
	OMPT_RTE_INTEL,
	OMPT_RTE_OMPSS,
	OMPT_UNKNOWN
} ompt_runtime_t;

void ompt_initialize(
	ompt_function_lookup_t lookup,
	const char *runtime_version_string, 
	unsigned ompt_version)
{
	ompt_runtime_t ompt_rte = OMPT_UNKNOWN;	
	int i;
	int r;

	UNREFERENCED_PARAMETER(ompt_version);

  if (ompt_enabled) 
	{
		Extrae_init();

#if defined(DEBUG) 
		printf("OMPT IS INITIALIZING: lookup functions with runtime version %s and ompt version %d\n",
		 runtime_version_string, ompt_version);
#endif

		if (strstr (runtime_version_string, "Intel") != NULL)
			ompt_rte = OMPT_RTE_INTEL;
		else if (strstr (runtime_version_string, "ibm") != NULL)
			ompt_rte = OMPT_RTE_IBM;
		else if (strstr (runtime_version_string, "nanos") != NULL)
			ompt_rte = OMPT_RTE_OMPSS;

#if defined(DEBUG)
		printf ("OMPTOOL: ompt_rte = %d\n", ompt_rte);
#endif 

		/* Ask OMPT for the routine ompt_set_callback */
		ompt_set_callback_fn = (int(*)(ompt_event_t, ompt_callback_t)) lookup("ompt_set_callback");
		assert (ompt_set_callback_fn != NULL);

		/* Ask OMPT for the routine ompt_get_thread_id */
		ompt_get_thread_id_fn = (ompt_thread_id_t(*)(void)) lookup("ompt_get_thread_id");
		assert (ompt_get_thread_id_fn != NULL);

#if defined(DEBUG)
		printf ("OMPTOOL: Recovered addresses for:\n");
		printf ("OMPTOOL: ompt_set_callback  = %p\n", ompt_set_callback_fn);
		printf ("OMPTOOL: ompt_get_thread_id = %p\n", ompt_get_thread_id_fn);
#endif

		i = 0;
		while (ompt_callbacks[i].evt != (ompt_event_t) 0)
		{
			/* What happens to IBM runtime? */
			if (ompt_rte == OMPT_RTE_IBM)
			{
				if (ompt_callbacks[i].evt != ompt_event_master_begin
				 && ompt_callbacks[i].evt != ompt_event_master_end)
				{
					r = ompt_set_callback_fn (ompt_callbacks[i].evt, ompt_callbacks[i].cbk);
#if defined(DEBUG)
					printf ("OMPTOOL: set_callback (%d) { %s } = %d\n", i, ompt_callbacks[i].evt_name, r);
#endif
				}
#if defined(DEBUG)
				else
				{
					printf ("OMPTOOL: Ignoring ompt_event_master_begin/end in IBM rte.\n");
				}
#endif
			}
			else
			{
				r = ompt_set_callback_fn (ompt_callbacks[i].evt, ompt_callbacks[i].cbk);
#if defined(DEBUG) 
				printf ("OMPTOOL: set_callback (%d) { %s } = %d\n", i, ompt_callbacks[i].evt_name, r);
#endif
			}
			i++;
		}

		if (getTrace_OMPLocks())
		{
#if defined(DEBUG)
			printf ("OMPTOOL: processing callbacks for locks\n");
#endif	
			i = 0;
			while (ompt_callbacks_locks[i].evt != (ompt_event_t) 0)
			{
				r = ompt_set_callback_fn (ompt_callbacks_locks[i].evt, ompt_callbacks_locks[i].cbk);
#if defined(DEBUG)
				printf ("OMPTOOL: set_callback (%d) { %s } = %d\n", i, ompt_callbacks_locks[i].evt_name, r);
#endif
				i++;
			}
		}
		else
		{
#if defined(DEBUG)
			printf ("OMPTOOL: NOT processing callbacks for locks\n");
#endif
		}

		Extrae_set_threadid_function (Extrae_OMPT_threadid);


		/* This point starts support for OMPT accelerators. */
		ompt_targets_initialized = ompt_target_initialize(lookup);

		UNREFERENCED_PARAMETER(r);
	}
}

/**
 * ompt_finalize
 *
 * Called from Backend_Finalize when there's support for OpenMP and OMPT enabled to 
 * make the final flush of the devices' buffers. The finalization has to be started 
 * by Extrae, because when the OmpSs/OMPT runtime finalizes, Extrae has already closed, 
 * and it is too late to flush the buffers.
 */
void ompt_finalize()
{
  /* Check whether the tracing for devices has been enabled */
  if (ompt_targets_initialized) 
  {
    ompt_target_finalize();
  }
} 

ompt_initialize_fn_t ompt_tool (void)
{
	return ompt_initialize;
}

