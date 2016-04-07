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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "file_set.h"
#include "object_tree.h"
#include "omp_prv_semantics.h"
#include "trace_to_prv.h"
#include "pthread_prv_events.h"
#include "semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "addresses.h"
#include "options.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#ifdef HAVE_BFD
# include "addr2info.h" 
#endif

#include "record.h"
#include "events.h"
#include "pthread_prv_events.h"

/******************************************************************************
 ***  WorkSharing_Event
 ******************************************************************************/

static int pthread_Call (event_t * current_event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset )
{
	unsigned int EvType, nEvType;
	unsigned long long EvValue, nEvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (current_event);
	EvValue = Get_EvValue (current_event);

	if (EvType == PTHREAD_RWLOCK_WR_EV || EvType == PTHREAD_RWLOCK_RD_EV ||
	    EvType == PTHREAD_RWLOCK_UNLOCK_EV ||
	    EvType == PTHREAD_MUTEX_LOCK_EV || EvType == PTHREAD_MUTEX_UNLOCK_EV ||
	    EvType == PTHREAD_COND_SIGNAL_EV || EvType == PTHREAD_COND_BROADCAST_EV ||
	    EvType == PTHREAD_COND_WAIT_EV || 
	    EvType == PTHREAD_BARRIER_WAIT_EV )
		Switch_State (STATE_SYNC, (EvValue != EVT_END), ptask, task, thread);
	else if (EvType == PTHREAD_EXIT_EV)
		Switch_State (STATE_RUNNING, (Get_EvValue (current_event) != EVT_BEGIN), ptask, task, thread);
	else 
		Switch_State (STATE_OVHD, (EvValue != EVT_END), ptask, task, thread);

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	if (EvType == PTHREAD_CREATE_EV)
	{
#if defined(HAVE_BFD)
		if (get_option_merge_SortAddresses() && EvValue > 0)
		{
			AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_FUNCTION);
			AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_LINE);
		}
#endif
		trace_paraver_event (cpu, ptask, task, thread, current_time, PTHREAD_FUNC_EV, EvValue);
		trace_paraver_event (cpu, ptask, task, thread, current_time, PTHREAD_FUNC_LINE_EV, EvValue);
	}

	Enable_pthread_Operation (EvType);
	if (EvType == PTHREAD_CREATE_EV)
		Translate_pthread_Operation (EvType, EvValue?1:0, &nEvType, &nEvValue);
	else
		Translate_pthread_Operation (EvType, EvValue, &nEvType, &nEvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time, nEvType, nEvValue);

	return 0;
}


static int pthread_Function_Event (event_t * current_event, 
	unsigned long long current_time, unsigned int cpu, unsigned int ptask,
	unsigned int task, unsigned int thread, FileSet_t *fset )
{
	unsigned int  EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvValue = Get_EvValue (current_event);

	Switch_State (STATE_RUNNING, (EvValue != EVT_END), ptask, task, thread);

#if defined(HAVE_BFD)
	if (get_option_merge_SortAddresses())
	{
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_FUNCTION);
		AddressCollector_Add (&CollectedAddresses, ptask, task, EvValue, ADDR2OMP_LINE);
	}
#endif

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	trace_paraver_event (cpu, ptask, task, thread, current_time, PTHREAD_FUNC_EV, EvValue);
	trace_paraver_event (cpu, ptask, task, thread, current_time, PTHREAD_FUNC_LINE_EV, EvValue);

	return 0;
}

SingleEv_Handler_t PRV_pthread_Event_Handlers[] = {
	{ PTHREAD_CREATE_EV, pthread_Call },
	{ PTHREAD_EXIT_EV, pthread_Call },
	{ PTHREAD_JOIN_EV, pthread_Call },
	{ PTHREAD_DETACH_EV, pthread_Call },
	{ PTHREAD_FUNC_EV, pthread_Function_Event },
	{ PTHREAD_RWLOCK_WR_EV, pthread_Call },
	{ PTHREAD_RWLOCK_RD_EV, pthread_Call },
	{ PTHREAD_RWLOCK_UNLOCK_EV, pthread_Call },
	{ PTHREAD_MUTEX_LOCK_EV, pthread_Call },
	{ PTHREAD_MUTEX_UNLOCK_EV, pthread_Call },
	{ PTHREAD_COND_SIGNAL_EV, pthread_Call },
	{ PTHREAD_COND_BROADCAST_EV, pthread_Call },
	{ PTHREAD_COND_WAIT_EV, pthread_Call },
	{ PTHREAD_BARRIER_WAIT_EV, pthread_Call },
	{ NULL_EV, NULL }
};

