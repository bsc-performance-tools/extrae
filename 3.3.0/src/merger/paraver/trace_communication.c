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

#include "timesync.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "object_tree.h"
#include "trace_communication.h"

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
#endif

/******************************************************************************
 ***  trace_communication
 ******************************************************************************/

void trace_communicationAt (unsigned ptask_s, unsigned task_s, unsigned thread_s, unsigned vthread_s,
	unsigned ptask_r, unsigned task_r, unsigned thread_r, unsigned vthread_r, event_t *send_begin,
	event_t *send_end, event_t *recv_begin, event_t *recv_end, 
	int atposition, off_t position)
{
	thread_t *thread_r_info, *thread_s_info;
	unsigned long long log_s, log_r, phy_s, phy_r;
	unsigned cpu_r, cpu_s;

	/* Look for the receive partner ... in the sender events */
	thread_r_info = GET_THREAD_INFO(ptask_r, task_r, thread_r);
	cpu_r = thread_r_info->cpu;

	/* Look for the sender partner ... in the receiver events */
	thread_s_info = GET_THREAD_INFO(ptask_s, task_s, thread_s);
	cpu_s = thread_s_info->cpu;

	/* Synchronize event times */
	log_s = TIMESYNC(ptask_s-1, task_s-1, Get_EvTime (send_begin));
	phy_s = TIMESYNC(ptask_s-1, task_s-1, Get_EvTime (send_end));
	log_r = TIMESYNC(ptask_r-1, task_r-1, Get_EvTime (recv_begin));
	phy_r = TIMESYNC(ptask_r-1, task_r-1, Get_EvTime (recv_end));

#if defined(DEBUG)
	fprintf (stderr, "trace_communicationAt: %u.%u.%u -> %u.%u.%u atposition=%d position=%llu\n",
	  ptask, task_s, thread_s, ptask, task_r, thread_r, atposition, position);
#endif

	trace_paraver_communication (cpu_s, ptask_s, task_s, thread_s, vthread_s, log_s, phy_s,
	  cpu_r, ptask_r, task_r, thread_r, vthread_r, log_r, phy_r, Get_EvSize (recv_end),
		Get_EvTag (recv_end), atposition, position);
}

#if defined(PARALLEL_MERGE)
int trace_pending_communication (unsigned int ptask_s, unsigned int task_s,
	unsigned int thread_s, unsigned vthread_s, event_t * begin_s, event_t * end_s, unsigned int ptask_r, unsigned int task_r)
{
	thread_t *thread_s_info = NULL;
	unsigned long long log_s, phy_s;
	unsigned cpu_s;

	thread_s_info = GET_THREAD_INFO(ptask_s, task_s, thread_s);
	cpu_s = thread_s_info->cpu; /* The receiver cpu is fixed at FixPendingCommunication */

	/* Synchronize event times */
	log_s = TIMESYNC (ptask_s-1, task_s-1, Get_EvTime (begin_s));
	phy_s = TIMESYNC (ptask_s-1, task_s-1, Get_EvTime (end_s));

	trace_paraver_pending_communication (cpu_s, ptask_s, task_s, thread_s, vthread_s, log_s,
		phy_s, task_r + 1, ptask_r, task_r + 1, thread_s /* 1? */ , thread_s /*vthread_r?*/,
		0ULL, 0ULL, Get_EvSize (begin_s), Get_EvTag (begin_s));

  return 0;
}
#endif

