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

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "pthread_probe.h"

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

static int TracePthreadLocks = FALSE;

void Extrae_pthread_instrument_locks (int value)
{
	TracePthreadLocks = value;
}

int Extrae_get_pthread_instrument_locks (void)
{
	return TracePthreadLocks;
}

void Probe_pthread_Create_Entry (void *p)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_CREATE_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_Create_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_CREATE_EV, EVT_END, EMPTY);
}

void Probe_pthread_Function_Entry (void *p)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_FUNC_EV, (UINT64)p ,EMPTY);
		Extrae_AnnotateCPU (LAST_READ_TIME);
	}
}

void Probe_pthread_Function_Exit (void)
{
	DEBUG
	if (mpitrace_on)
	{
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_FUNC_EV, EVT_END ,EMPTY);
		Extrae_AnnotateCPU (LAST_READ_TIME);
	}
}

void Probe_pthread_Exit_Entry(void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_EXIT_EV, EVT_BEGIN, EMPTY);
}

void Probe_pthread_Join_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_JOIN_EV, EVT_BEGIN, EMPTY);
}

void Probe_pthread_Join_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_JOIN_EV, EVT_END, EMPTY);
}

void Probe_pthread_Detach_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_DETACH_EV, EVT_BEGIN, EMPTY);
}

void Probe_pthread_Detach_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_DETACH_EV, EVT_END, EMPTY);
}

/* RW locks */

void Probe_pthread_rwlock_lockwr_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_RWLOCK_WR_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_rwlock_lockwr_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_RWLOCK_WR_EV, EMPTY, EMPTY);
}

void Probe_pthread_rwlock_lockrd_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_RWLOCK_RD_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_rwlock_lockrd_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_RWLOCK_RD_EV, EMPTY, EMPTY);
}

void Probe_pthread_rwlock_unlock_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_RWLOCK_UNLOCK_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_rwlock_unlock_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_RWLOCK_UNLOCK_EV, EMPTY, EMPTY);
}

/* Mutex locks */

void Probe_pthread_mutex_lock_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_MUTEX_LOCK_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_mutex_lock_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_MUTEX_LOCK_EV, EMPTY, EMPTY);
}

void Probe_pthread_mutex_unlock_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_MUTEX_UNLOCK_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_mutex_unlock_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_MUTEX_UNLOCK_EV, EMPTY, EMPTY);
}

/* CONDs */

void Probe_pthread_cond_signal_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_COND_SIGNAL_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_cond_signal_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_COND_SIGNAL_EV, EMPTY, EMPTY);
}

void Probe_pthread_cond_broadcast_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_COND_BROADCAST_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_cond_broadcast_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_COND_BROADCAST_EV, EMPTY, EMPTY);
}

void Probe_pthread_cond_wait_Entry (void *p)
{
	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_COND_WAIT_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_cond_wait_Exit (void *p)
{
	UNREFERENCED_PARAMETER(p);

	DEBUG
	if (mpitrace_on && TracePthreadLocks)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_COND_WAIT_EV, EMPTY, EMPTY);
}

void Probe_pthread_Barrier_Wait_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(LAST_READ_TIME, PTHREAD_BARRIER_WAIT_EV, EVT_BEGIN, EMPTY);
}

void Probe_pthread_Barrier_Wait_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_PTHEVENTANDCOUNTERS(TIME, PTHREAD_BARRIER_WAIT_EV, EVT_END, EMPTY);
}
