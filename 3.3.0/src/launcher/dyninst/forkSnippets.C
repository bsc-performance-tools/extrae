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

#include <BPatch.h>
#include <BPatch_point.h>

#include "commonSnippets.h"
#include "forkSnippets.h"

void InstrumentForks (BPatch_image *appImage)
{
	wrapRoutine (appImage, "fork", "Extrae_Probe_fork_Entry", "Extrae_Probe_fork_Exit");
	wrapRoutine (appImage, "wait", "Extrae_Probe_wait_Entry", "Extrae_Probe_wait_Exit");
	wrapRoutine (appImage, "waitpid", "Extrae_Probe_waitpid_Entry", "Extrae_Probe_waitpid_Exit");
	wrapRoutine (appImage, "system", "Extrae_Probe_system_Entry", "Extrae_Probe_system_Exit", 1);

	wrapRoutine (appImage, "execl", "Extrae_Probe_exec_l_Entry", "", 1);
	wrapRoutine (appImage, "execle", "Extrae_Probe_exec_l_Entry", "", 1);
	wrapRoutine (appImage, "execlp", "Extrae_Probe_exec_l_Entry", "", 1);

	wrapRoutine (appImage, "execv", "Extrae_Probe_exec_v_Entry", "", 2);
	// wrapRoutine (appImage, "execve", "Extrae_Probe_exec_v_Entry", "", 2);  // not in all systems?
	wrapRoutine (appImage, "execvp", "Extrae_Probe_exec_v_Entry", "", 2);
	wrapRoutine (appImage, "execvpe", "Extrae_Probe_exec_v_Entry", "", 2);
}

