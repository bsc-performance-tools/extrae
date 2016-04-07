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

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "fork_probe.h"
#include "misc_wrapper.h"
#include "sampling-timer.h"

static pid_t MYPID; /* Used to determin parent's PID and discern between parent & child */
static int IamMasterOfAllProcesses = TRUE;
static int MyDepthOfAllProcesses = 1;

int Extrae_isProcessMaster (void)
{
	return IamMasterOfAllProcesses;
}

int Extrae_myDepthOfAllProcesses (void)
{
	return MyDepthOfAllProcesses;
}

void Extrae_Probe_fork_Entry (void)
{
	MYPID = getpid();

	Backend_Enter_Instrumentation (2);
	Probe_fork_Entry ();

	unsetTimeSampling();

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
	{
		Extrae_init_tracing (TRUE);
	}
}

void Extrae_Probe_fork_Exit (void)
{
	if (MYPID != getpid())
	{
		/* If I'm the child, I'm no longer the master */
		IamMasterOfAllProcesses = FALSE;

		/* Increase depth of this process */
		MyDepthOfAllProcesses++;

		Extrae_Probe_fork_child_Exit();
	}
	else
		Extrae_Probe_fork_parent_Exit();

	setTimeSampling_postfork ();
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

/* This is not safest, but should work to identify all the system calls */
static extrae_value_t last_system_id = 1;

void Extrae_Probe_system_Entry (char *newbinary)
{
	Backend_Enter_Instrumentation (2);
	Probe_system_Entry();

	/* Dude!, grab the binary we're about to execute and post it into the .sym file */
	Extrae_define_event_type_Wrapper (SYSTEM_BIN_EV, "system() binary name", 1,
		&last_system_id, &newbinary);
	TRACE_MISCEVENT(LAST_READ_TIME, USER_EV, SYSTEM_BIN_EV, last_system_id);

	last_system_id++;
}

void Extrae_Probe_system_Exit (void)
{
	Probe_system_Exit();
	Backend_Leave_Instrumentation();
}

void Extrae_Probe_exec_l_Entry (char *newbinary)
{
	printf ("Extrae_Probe_exec_l_Entry, Extrae_Probe_exec_l_Entry, Extrae_Probe_exec_l_Entry\n");

	Backend_Enter_Instrumentation (2);
	Probe_exec_Entry();

	extrae_value_t v = getpid();

	/* Dude!, we are changing the process image! Dump all the information
	   generated into the tracefile */
	Extrae_define_event_type_Wrapper (EXEC_BIN_EV, "exec() binary name", 1, &v, &newbinary);
	TRACE_MISCEVENT(LAST_READ_TIME, USER_EV, EXEC_BIN_EV, getpid());

	Extrae_fini_Wrapper();
}

void Extrae_Probe_exec_v_Entry (char *newbinary, char *const argv[])
{
	#define BUFFER_SIZE 1024
	char buffer[BUFFER_SIZE];
	char *pbuffer[1] = { buffer };
	int i = 0;
	int remaining = BUFFER_SIZE -1;
	int position = 0;

	UNREFERENCED_PARAMETER(newbinary);

	Backend_Enter_Instrumentation (2);
	Probe_exec_Entry();

	for (i = 0; i < BUFFER_SIZE; i++)
		buffer[i] = 0;

	i = 0;
	while (argv[i] != NULL && remaining > 0)
	{
		int length = strlen (argv[i]);

		if (length < remaining)
		{
			strncpy (&buffer[position], argv[i], length);
			buffer[position+length] = ' ';
			position += length + 1;
			remaining -= length + 1;
		}
		else
		{
			strncpy (&buffer[position], argv[i], remaining);
			remaining = 0;
		}
		i++;
	}

	extrae_value_t v = getpid();

	/* Dude!, we are changing the process image! Dump all the information
	   generated into the tracefile */
	pbuffer[0] = buffer;
	Extrae_define_event_type_Wrapper (EXEC_BIN_EV, "exec() binary name", 1, &v, pbuffer);
	TRACE_MISCEVENT(LAST_READ_TIME, USER_EV, EXEC_BIN_EV, getpid());

	Extrae_fini_Wrapper();

	#undef BUFFER_SIZE
}

void Extrae_Probe_exec_Exit (void)
{
	Probe_exec_Exit();
	Backend_Leave_Instrumentation();
}

