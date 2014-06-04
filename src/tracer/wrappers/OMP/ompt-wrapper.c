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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.5/src/tracer/wrappers/OMP/omp_wrapper.c $
 | @last_commit: $Date: 2014-02-20 16:48:43 +0100 (jue, 20 feb 2014) $
 | @version:     $Revision: 2487 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: omp_wrapper.c 2487 2014-02-20 15:48:43Z harald $";

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
#ifdef HAVE_ASSERT_H
# include <assert.h>
#endif
#ifdef HAVE_DFLCN_H
# define __USE_GNU
#  include <dlfcn.h>
# undef __USE_GNU
#endif

#include "omp-common.h"

#include "ompt-helper.h"

// #define DEBUG

#if defined(__INTEL_COMPILER)
# define ompt_interface_fn(f) OMPT_API_FNTYPE(f) f ## _fn;
#  include "/home/bsc41/bsc41273/src/ompt/ompt-intel-openmp/itt/libomp_oss/src/ompt-fns.h"
# undef ompt_interface_fn
#endif

//*****************************************************************************
// interface operations
//*****************************************************************************

int (*ompt_set_callback_fn)(ompt_event_t, ompt_callback_t) = NULL;
ompt_thread_id_t (*ompt_get_thread_id_fn)(void) = NULL;
ompt_frame_t* (*ompt_get_task_frame_fn)(int) = NULL;

static ompt_thread_id_t *ompt_thids = NULL;
static unsigned n_ompt_thids = 0;
static pthread_mutex_t mutex_thids = PTHREAD_MUTEX_INITIALIZER;

void Extrae_OMPT_register_ompt_thread_id (ompt_thread_id_t ompt_thid)
{
	pthread_mutex_lock (&mutex_thids);
	ompt_thids = (ompt_thread_id_t*) realloc (ompt_thids,
	  (n_ompt_thids+1)*sizeof(ompt_thread_id_t));
	assert (ompt_thids != NULL);

	ompt_thids[n_ompt_thids] = ompt_thid;
	n_ompt_thids++;
	pthread_mutex_unlock (&mutex_thids);
}

unsigned Extrae_OMPT_threadid (void)
{
	ompt_thread_id_t thd = ompt_get_thread_id_fn();
	unsigned u;

	pthread_mutex_lock (&mutex_thids);
	for (u = 0; u < n_ompt_thids; u++)
		if (ompt_thids[u] == thd)
		{
			pthread_mutex_unlock (&mutex_thids);
			return u;
		}

	pthread_mutex_unlock (&mutex_thids);
	assert (1 != 1);
	return 0;
}


#if defined(DEBUG)
# define PROTOTYPE_MESSAGE_NOTHREAD(fmt, ...) \
   printf ("THREAD=??/?? TIME=%llu %s" fmt "\n", \
    0ull, \
    __func__, \
    ## __VA_ARGS__)
# if defined(__INTEL_COMPILER)
#  define PROTOTYPE_MESSAGE(fmt, ...) \
    printf ("THREAD=%u/%llu TIME=%llu %s" fmt "\n", \
     0u, \
     0ull, \
     0ull, \
     __func__, \
     ## __VA_ARGS__)
# else
#  define PROTOTYPE_MESSAGE(fmt, ...) \
    printf ("THREAD=%u/%llu TIME=%llu %s" fmt "\n", \
     Extrae_OMPT_threadid(), \
     ompt_get_thread_id_fn(), \
     0, \
     __func__, \
     ## __VA_ARGS__)
# endif
#else
# define PROTOTYPE_MESSAGE(...)
# define PROTOTYPE_MESSAGE_NOTHREAD(...)
#endif

void OMPT_event_initial_thread_begin (ompt_thread_id_t thid)
{
	PROTOTYPE_MESSAGE_NOTHREAD(" TYPE %d THID %ld", 0, thid);
	Extrae_OMPT_register_ompt_thread_id (thid);
	// TODO: EXTRAE
}

void OMPT_event_thread_begin (ompt_thread_type_t type, ompt_thread_id_t thid)
{
	UNREFERENCED_PARAMETER(type);

	PROTOTYPE_MESSAGE_NOTHREAD(" TYPE %d (worker == %d) THID %ld", type, ompt_thread_worker, thid);
	Extrae_OMPT_register_ompt_thread_id (thid);
	// TODO: EXTRAE
}

void OMPT_event_thread_end (ompt_thread_type_t type, ompt_thread_id_t thid)
{
	UNREFERENCED_PARAMETER(type);

	PROTOTYPE_MESSAGE(" TYPE %d THID %ld", type, thid);
	// TODO: EXTRAE
}

void OMPT_event_loop_begin (ompt_parallel_id_t pid, ompt_task_id_t tid, void *wf)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Loop_Entry ();
}

void OMPT_event_loop_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid);
	Extrae_OMPT_Loop_Exit ();
}

void OMPT_event_parallel_begin (ompt_task_id_t tid, ompt_frame_t *ptf, ompt_parallel_id_t pid, uint32_t req_team_size, void *pf)
{
	UNREFERENCED_PARAMETER(ptf);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(req_team_size);

	PROTOTYPE_MESSAGE(" (%ld, %p, %ld, %u, %p)", tid, ptf, pid, req_team_size, pf);
	Extrae_OMPT_register_ompt_parallel_id_pf (pid, pf);
	Extrae_OpenMP_ParRegion_Entry ();
}

void OMPT_event_parallel_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_unregister_ompt_parallel_id_pf (pid);
	Extrae_OpenMP_ParRegion_Exit();
}

void OMPT_event_barrier_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Barrier_Entry ();
}

void OMPT_event_barrier_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Barrier_Exit ();
}

void OMPT_event_wait_barrier_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_wait_barrier_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_master_begin(ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Master_Entry ();
}

void OMPT_event_master_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_Master_Exit ();
}

void OMPT_event_sections_begin (ompt_parallel_id_t pid, ompt_task_id_t tid, void *wf)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Sections_Entry();
}

void OMPT_event_sections_end (ompt_parallel_id_t pid, ompt_task_id_t tid, void *wf)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Sections_Exit();
}

void OMPT_event_single_others_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE, nothing atm?
}

void OMPT_event_single_others_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE, nothing atm?
}

void OMPT_event_single_in_block_begin (ompt_parallel_id_t pid, ompt_task_id_t tid, void *wf)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Single_Entry ();
}

void OMPT_event_single_in_block_end (ompt_parallel_id_t pid, ompt_task_id_t tid, void *wf)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Single_Exit ();
}

void OMPT_event_task_begin (ompt_task_id_t ptid, ompt_frame_t *ptf, ompt_task_id_t tid, void *ntf)
{
	UNREFERENCED_PARAMETER(ptid);
	UNREFERENCED_PARAMETER(ptf);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %p, %ld, %p)", ptid, ptf, tid, ntf);
	Extrae_OMPT_register_ompt_task_id_tf (tid, ntf, FALSE);
	//Extrae_OpenMP_TaskUF_Entry (ntf); NOTE: Task does not start running here (HSG, for IBM)!
}

void OMPT_event_task_end (ompt_task_id_t ptid, ompt_frame_t *ptf, ompt_task_id_t tid, void *ntf)
{
	UNREFERENCED_PARAMETER(ptid);
	UNREFERENCED_PARAMETER(ptf);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(ntf);

	PROTOTYPE_MESSAGE(" (%ld, %p, %ld, %p)", ptid, ptf, tid, ntf);
	Extrae_OpenMP_TaskUF_Exit (); // NOTE: Task does not start running here (HSG, for IBM)!
	Extrae_OMPT_unregister_ompt_task_id_tf (tid);
}

void OMPT_event_wait_taskwait_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_wait_taskwait_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_taskwait_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Taskwait_Entry ();
}

void OMPT_event_taskwait_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OpenMP_Taskwait_Exit ();
}

void OMPT_event_taskgroup_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_taskgroup_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_wait_taskgroup_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_wait_taskgroup_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	// TODO: EXTRAE
}

void OMPT_event_workshare_begin (ompt_parallel_id_t pid, ompt_task_id_t tid, void *wf)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Workshare_Entry();
}

void OMPT_event_workshare_end (ompt_parallel_id_t pid, ompt_task_id_t tid, void *wf)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);
	UNREFERENCED_PARAMETER(wf);

	PROTOTYPE_MESSAGE(" (%ld, %ld, %p)", pid, tid, wf);
	Extrae_OMPT_Workshare_Exit();
}

void OMPT_event_idle_begin (ompt_thread_id_t thid)
{
	PROTOTYPE_MESSAGE(" (THID = %ld)", thid);
	// TODO: EXTRAE
}

void OMPT_event_idle_end (ompt_thread_id_t thid)
{
	PROTOTYPE_MESSAGE(" (THID = %ld)", thid);
	// TODO: EXTRAE
}

void OMPT_event_release_lock (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Unlock_Exit();
}

void OMPT_event_release_nest_lock_last (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Unlock_Exit();
}

void OMPT_event_release_critical (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Unlock_Exit();
	Extrae_OMPT_Critical_Exit();
}

void OMPT_event_release_ordered (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Unlock_Exit ();
}

void OMPT_event_release_atomic (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Unlock_Exit ();
	Extrae_OMPT_Atomic_Exit();
}

void OMPT_event_implicit_task_begin (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_register_ompt_task_id_tf (tid, Extrae_OMPT_get_pf_parallel_id(pid), TRUE);
	Extrae_OpenMP_UF_Entry (Extrae_OMPT_get_pf_parallel_id(pid));
}

void OMPT_event_implicit_task_end (ompt_parallel_id_t pid, ompt_task_id_t tid)
{
	UNREFERENCED_PARAMETER(pid);
	UNREFERENCED_PARAMETER(tid);

	PROTOTYPE_MESSAGE(" (%ld, %ld)", pid, tid);
	Extrae_OMPT_unregister_ompt_task_id_tf (tid);
	Extrae_OpenMP_UF_Exit ();
}

void OMPT_event_initial_task_begin (ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld)", tid);
	// TODO: EXTRAE
}

void OMPT_event_initial_task_end (ompt_task_id_t tid)
{
	PROTOTYPE_MESSAGE(" (%ld)", tid);
	// TODO: EXTRAE
}

void OMPT_event_task_switch (ompt_task_id_t stid, ompt_task_id_t rtid)
{
	void *tf;

	/* This event denotes that the previous task was suspended and another resumed */
	PROTOTYPE_MESSAGE(" (%ld, %ld)", stid, rtid);

	/* Leave a task function if it's not the implicit task. The implicit
	   task is automatically instrumented elsewhere */
	if ((tf = Extrae_OMPT_get_tf_task_id(stid)) &&
	     !Extrae_OMPT_get_tf_task_id_is_implicit(stid))
		Extrae_OpenMP_TaskUF_Exit ();

	/* Enter a task function if it's not the implicit task. The implicit
	   task is automatically instrumented elsewhere */
	if ((tf = Extrae_OMPT_get_tf_task_id (rtid)) &&
	     !Extrae_OMPT_get_tf_task_id_is_implicit(rtid))
		Extrae_OpenMP_TaskUF_Entry (tf);
}

void OMPT_event_wait_lock (ompt_wait_id_t wid)
{
	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Entry ();
}

void OMPT_event_acquired_lock (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Exit ((void*)wid);
}

void OMPT_event_wait_nest_lock (ompt_wait_id_t wid)
{
	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Entry ();
}

void OMPT_event_acquired_nest_lock_first (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Named_Lock_Exit ((void*)wid);
}

void OMPT_event_release_nest_lock_prev (ompt_wait_id_t wid)
{
	PROTOTYPE_MESSAGE(" (%ld)", wid);
	// TODO: EXTRAE
}

void OMPT_event_acquired_nest_lock_next (ompt_wait_id_t wid)
{
	PROTOTYPE_MESSAGE(" (%ld)", wid);
	// TODO: EXTRAE
}

void OMPT_event_wait_critical (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Entry ();
}

void OMPT_event_acquired_critical (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Exit ();
	Extrae_OMPT_Critical_Entry();
}

void OMPT_event_wait_ordered (ompt_wait_id_t wid)
{
	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Entry ();
}

void OMPT_event_acquired_ordered (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Exit ();
}

void OMPT_event_wait_atomic (ompt_wait_id_t wid)
{
	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Entry ();
}

void OMPT_event_acquired_atomic (ompt_wait_id_t wid)
{
	UNREFERENCED_PARAMETER(wid);

	PROTOTYPE_MESSAGE(" (%ld)", wid);
	Extrae_OpenMP_Unnamed_Lock_Exit ();
	Extrae_OMPT_Atomic_Entry();
}

void OMPT_event_control (uint64_t command, uint64_t modifier)
{
	PROTOTYPE_MESSAGE(" (cmd = %lu, mod = %lu)", command, modifier);

	if (command == 1) /* API spec: start or restart monitoring */
		Extrae_restart();
	else if (command == 2) /* API spec: pause monitoring */
		Extrae_shutdown();
	else if (command == 3) /* API spec: flush tool buffer & continue */
		Extrae_flush();
	else if (command == 4) /* API spec: shutdown */
		Extrae_fini();
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

struct OMPT_callbacks_st ompt_callbacks[] =
{
	 CALLBACK_ENTRY (ompt_event_loop_begin, OMPT_event_loop_begin), /* 0 */
	 CALLBACK_ENTRY (ompt_event_loop_end, OMPT_event_loop_end),
	 CALLBACK_ENTRY (ompt_event_parallel_begin, OMPT_event_parallel_begin),
	 CALLBACK_ENTRY (ompt_event_parallel_end, OMPT_event_parallel_end),
	 CALLBACK_ENTRY (ompt_event_barrier_begin, OMPT_event_barrier_begin),
	 CALLBACK_ENTRY (ompt_event_barrier_end, OMPT_event_barrier_end),
	 CALLBACK_ENTRY (ompt_event_wait_barrier_begin, OMPT_event_wait_barrier_begin),
	 CALLBACK_ENTRY (ompt_event_wait_barrier_end, OMPT_event_wait_barrier_end),
	 CALLBACK_ENTRY (ompt_event_master_begin, OMPT_event_master_begin),
	 CALLBACK_ENTRY (ompt_event_master_end, OMPT_event_master_end),
	 CALLBACK_ENTRY (ompt_event_sections_begin, OMPT_event_sections_begin), /* 10 */
	 CALLBACK_ENTRY (ompt_event_sections_end, OMPT_event_sections_end),
	 CALLBACK_ENTRY (ompt_event_single_others_begin, OMPT_event_single_others_begin),
	 CALLBACK_ENTRY (ompt_event_single_others_end, OMPT_event_single_others_end),
	 CALLBACK_ENTRY (ompt_event_single_in_block_begin, OMPT_event_single_in_block_begin),
	 CALLBACK_ENTRY (ompt_event_single_in_block_end, OMPT_event_single_in_block_end),
	 CALLBACK_ENTRY (ompt_event_task_begin, OMPT_event_task_begin),
	 CALLBACK_ENTRY (ompt_event_task_end, OMPT_event_task_end),
	 CALLBACK_ENTRY (ompt_event_wait_taskwait_begin, OMPT_event_wait_taskwait_begin),
	 CALLBACK_ENTRY (ompt_event_wait_taskwait_end, OMPT_event_wait_taskwait_end),
	 CALLBACK_ENTRY (ompt_event_taskwait_begin, OMPT_event_taskwait_begin), /* 20 */
	 CALLBACK_ENTRY (ompt_event_taskwait_end, OMPT_event_taskwait_end),
	 CALLBACK_ENTRY (ompt_event_wait_taskgroup_begin, OMPT_event_wait_taskgroup_begin),
	 CALLBACK_ENTRY (ompt_event_wait_taskgroup_end, OMPT_event_wait_taskgroup_end),
	 CALLBACK_ENTRY (ompt_event_taskgroup_begin, OMPT_event_taskgroup_begin),
	 CALLBACK_ENTRY (ompt_event_taskgroup_end, OMPT_event_taskgroup_end),
	 CALLBACK_ENTRY (ompt_event_workshare_begin, OMPT_event_workshare_begin),
	 CALLBACK_ENTRY (ompt_event_workshare_end, OMPT_event_workshare_end),
	 CALLBACK_ENTRY (ompt_event_idle_begin, OMPT_event_idle_begin),
	 CALLBACK_ENTRY (ompt_event_idle_end, OMPT_event_idle_end),
	 CALLBACK_ENTRY (ompt_event_release_lock, OMPT_event_release_lock), /* 30 */
	 CALLBACK_ENTRY (ompt_event_release_nest_lock_last, OMPT_event_release_nest_lock_last),
	 CALLBACK_ENTRY (ompt_event_release_critical, OMPT_event_release_critical),
	 CALLBACK_ENTRY (ompt_event_release_ordered, OMPT_event_release_ordered),
	 CALLBACK_ENTRY (ompt_event_release_atomic, OMPT_event_release_atomic),
	 CALLBACK_ENTRY (ompt_event_implicit_task_begin, OMPT_event_implicit_task_begin),
	 CALLBACK_ENTRY (ompt_event_implicit_task_end, OMPT_event_implicit_task_end),
	 CALLBACK_ENTRY (ompt_event_initial_task_begin, OMPT_event_initial_task_begin),
	 CALLBACK_ENTRY (ompt_event_initial_task_end, OMPT_event_initial_task_end),
	 CALLBACK_ENTRY (ompt_event_task_switch, OMPT_event_task_switch),
	 CALLBACK_ENTRY (ompt_event_wait_lock, OMPT_event_wait_lock), /* 40 */
	 CALLBACK_ENTRY (ompt_event_acquired_lock, OMPT_event_acquired_lock),
	 CALLBACK_ENTRY (ompt_event_wait_nest_lock, OMPT_event_wait_nest_lock),
	 CALLBACK_ENTRY (ompt_event_acquired_nest_lock_first, OMPT_event_acquired_nest_lock_first),
	 CALLBACK_ENTRY (ompt_event_release_nest_lock_prev, OMPT_event_release_nest_lock_prev),
	 CALLBACK_ENTRY (ompt_event_acquired_nest_lock_next, OMPT_event_acquired_nest_lock_next),
	 CALLBACK_ENTRY (ompt_event_wait_critical, OMPT_event_wait_critical),
	 CALLBACK_ENTRY (ompt_event_acquired_critical, OMPT_event_acquired_critical),
	 CALLBACK_ENTRY (ompt_event_wait_ordered, OMPT_event_wait_ordered),
	 CALLBACK_ENTRY (ompt_event_acquired_ordered, OMPT_event_acquired_ordered),
	 CALLBACK_ENTRY (ompt_event_wait_atomic, OMPT_event_wait_atomic), /* 50 */
	 CALLBACK_ENTRY (ompt_event_acquired_atomic, OMPT_event_acquired_atomic),
	 CALLBACK_ENTRY (ompt_event_thread_begin, OMPT_event_thread_begin),
	 CALLBACK_ENTRY (ompt_event_thread_end, OMPT_event_thread_end),
	 CALLBACK_ENTRY (ompt_event_control, OMPT_event_control),
	{ "empty,", (ompt_event_t) 0, 0 },
};

typedef enum { OMPT_RTE_IBM, OMPT_RTE_INTEL, OMPT_UNKNOWN } ompt_runtime_t;

int ompt_initialize(
	ompt_function_lookup_t lookup,
	const char *runtime_version_string, 
	int ompt_version)
{
	ompt_runtime_t ompt_rte = OMPT_UNKNOWN;	
	int i;
	int r;

#if defined(DEBUG) 
	printf("OMPT IS INITIALIZING: lookup functions with runtime version %s and ompt version %d\n",
	  runtime_version_string, ompt_version);
#endif

	if (strstr (runtime_version_string, "Intel") != NULL)
		ompt_rte = OMPT_RTE_INTEL;
	else if (strstr (runtime_version_string, "ibm") != NULL)
		ompt_rte = OMPT_RTE_IBM;

#if defined(DEBUG)
	printf ("OMPTOOL: ompt_rte = %d\n", ompt_rte);
#endif

	ompt_set_callback_fn = (int(*)(ompt_event_t, ompt_callback_t)) lookup("ompt_set_callback");
	assert (ompt_set_callback_fn != NULL);

	ompt_get_thread_id_fn = (ompt_thread_id_t(*)(void)) lookup("ompt_get_thread_id");
	assert (ompt_get_thread_id_fn != NULL);

	ompt_get_task_frame_fn = (ompt_frame_t*(*)(int)) lookup ("ompt_get_task_frame");
	assert (ompt_get_task_frame_fn != NULL);

#if defined(DEBUG)
	printf ("OMPTOOL: Recovered addresses for:\n");
	printf ("OMPTOOL: ompt_set_callback  = %p\n", ompt_set_callback_fn);
	printf ("OMPTOOL: ompt_get_thread_id = %p\n", ompt_get_thread_id_fn);
	printf ("OMPTOOL: ompt_get_task_frame_fn = %p\n", ompt_get_task_frame_fn);
#endif

	i = 0;
	while (ompt_callbacks[i].evt != (ompt_event_t) 0)
	{
		if (ompt_rte == OMPT_RTE_IBM &&
			(ompt_callbacks[i].evt != ompt_event_master_begin &&
			 ompt_callbacks[i].evt != ompt_event_master_end))
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
		i++;
	}

	Extrae_set_threadid_function (Extrae_OMPT_threadid);

	return 1;
}

