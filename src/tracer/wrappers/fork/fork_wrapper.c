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

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "fork_probe.h"
#include "misc_wrapper.h"

static pid_t MYPID; /* Used to determin parent's PID and discern between parent & child */

void Extrae_Probe_fork_Entry (void)
{
	MYPID = getpid();

	Backend_Enter_Instrumentation (2);
	Probe_fork_Entry ();

#if USE_HARDWARE_COUNTERS
	/* We need to stop counters and restart after fork() */
	HWC_Stop_Current_Set (LAST_READ_TIME, THREADID);
#endif
}

void Extrae_Probe_fork_parent_Exit (void)
{
	Probe_fork_parent_Exit();

#if USE_HARDWARE_COUNTERS
	/* We need to start counters again */
	HWC_Start_Current_Set (0, LAST_READ_TIME, THREADID);
#endif
	Backend_Leave_Instrumentation();
}

void Extrae_Probe_fork_child_Exit (void)
{
	if (mpitrace_on)
		Extrae_init_tracing (TRUE);
}

void Extrae_Probe_fork_Exit (void)
{
	if (MYPID == getpid())
		Extrae_Probe_fork_parent_Exit();
	else
		Extrae_Probe_fork_child_Exit();
}

void Extrae_Probe_wait_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_wait_Entry();
}

void Extrae_Probe_wait_Exit (void)
{
	Probe_wait_Exit();
	Backend_Leave_Instrumentation();
}

void Extrae_Probe_waitpid_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_waitpid_Entry();
}

void Extrae_Probe_waitpid_Exit (void)
{
	Probe_waitpid_Exit();
	Backend_Leave_Instrumentation();
}

void Extrae_Probe_exec_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_exec_Entry();
}

void Extrae_Probe_exec_Exit (void)
{
	Probe_exec_Exit();
	Backend_Leave_Instrumentation();
}

