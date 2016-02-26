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

#if HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "fork_probe.h"

#if 0
# define DEBUG fprintf (stdout, "PID: %d THREAD %d: %s\n", getpid(), THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

void Probe_fork_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS (LAST_READ_TIME, FORK_EV, EVT_BEGIN, TRUE);
}

void Probe_fork_parent_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENT (TIME, FORK_EV, EVT_END);
}

void Probe_wait_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, WAIT_EV, EVT_BEGIN, TRUE);
}

void Probe_wait_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(TIME, WAIT_EV, EVT_END, TRUE);
}

void Probe_waitpid_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, WAITPID_EV, EVT_BEGIN, TRUE);
}

void Probe_waitpid_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(TIME, WAITPID_EV, EVT_END, TRUE);
}

void Probe_exec_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, EXEC_EV, EVT_BEGIN, TRUE);
}

void Probe_exec_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(TIME, EXEC_EV, EVT_END, TRUE);
}

void Probe_system_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, SYSTEM_EV, EVT_BEGIN, TRUE);
}

void Probe_system_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_EVENTANDCOUNTERS(TIME, SYSTEM_EV, EVT_END, TRUE);
}



