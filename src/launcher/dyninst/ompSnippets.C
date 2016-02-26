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

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <list>
#include <string>
#include <iostream>
#include <fstream>

using namespace std; 

#include "commonSnippets.h"
#include "ompSnippets.h"

#include <BPatch_function.h>

void InstrumentOMPruntime_Intel (BPatch_image *appImage, BPatch_process *appProcess)
{
	/* Gather information for all the instrumented calls */
	BPatch_function *function = getRoutine (
		"Extrae_intel_kmpc_runtime_init_dyninst", appImage, false);
	if (function == NULL)
	{
		cerr << PACKAGE_NAME << ": Fatal error! Cannot find Extrae_intel_kmpc_runtime_init_dyninst! Dying..." << endl;
		exit (-1);
	}

	BPatch_Vector<BPatch_snippet*> args;
	BPatch_function *pfunction = getRoutine (
		"__kmpc_fork_call", appImage, false);
	if (pfunction == NULL)
	{
		cerr << PACKAGE_NAME << ": Fatal error! Cannot find __kmpc_fork_call! Dying..." << endl;
		exit (-1);
	}
	BPatch_constExpr p0 = pfunction->getBaseAddr();
	args.push_back (&p0);

	BPatch_funcCallExpr s (*function, args);

	appProcess->oneTimeCode (s);
}


void InstrumentOMPruntime (bool locks, ApplicationType *at, BPatch_image *appImage)
{
	if (at->get_OpenMP_rte() == ApplicationType::Intel_v8_1 ||
	    at->get_OpenMP_rte() == ApplicationType::Intel_v9_1 ||
	    at->get_OpenMP_rte() == ApplicationType::Intel_v11)
	{
		/* kmpc_fork_call is handled differently in v11, for v11 look at
		   InstrumentCalls in extrae.C */
		if (!at->get_OpenMP_rte() == ApplicationType::Intel_v11)
		{
			wrapRoutine (appImage, "__kmpc_fork_call",
			  "Extrae_OpenMP_ParRegion_Entry",
			  "Extrae_OpenMP_ParRegion_Exit");
		}

		wrapRoutine (appImage, "__kmpc_barrier",
		  "Extrae_OpenMP_Barrier_Entry",
		  "Extrae_OpenMP_Barrier_Exit");
		wrapRoutine (appImage, "__kmpc_invoke_task_func",
		  "Extrae_OpenMP_ParDO_Entry",
			"Extrae_OpenMP_ParDO_Exit");
		if (locks)
		{
			wrapRoutine (appImage, "__kmpc_set_lock",
			  "Extrae_OpenMP_Unnamed_Lock_Entry",
			  "Extrae_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, "__kmpc_unset_lock",
			  "Extrae_OpenMP_Unnamed_Unlock_Entry",
			  "Extrae_OpenMP_Unnamed_Unlock_Exit");
			wrapRoutine (appImage, "__kmpc_critical",
			  "Extrae_OpenMP_Named_Lock_Entry",
			  "Extrae_OpenMP_Named_Lock_Exit");
			wrapRoutine (appImage, "__kmpc_end_critical",
			  "Extrae_OpenMP_Named_Unlock_Entry",
				"Extrae_OpenMP_Named_Unlock_Exit");
		}
	}

	if (at->get_OpenMP_rte() == ApplicationType::IBM_v16)
	{
		wrapRoutine (appImage, "_xlsmpParallelDoSetup_TPO",
		  "Extrae_OpenMP_ParDO_Entry",
		  "Extrae_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, "_xlsmpWSDoSetup_TPO",
		  "Extrae_OpenMP_DO_Entry",
			"Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "_xlsmpParRegionSetup_TPO",
		  "Extrae_OpenMP_ParRegion_Entry",
			"Extrae_OpenMP_ParRegion_Exit");
		wrapRoutine (appImage, "_xlsmpWSSectSetup_TPO",
		  "Extrae_OpenMP_Section_Entry",
		  "Extrae_OpenMP_Section_Exit");
		wrapRoutine (appImage, "_xlsmpSingleSetup_TPO",
		  "Extrae_OpenMP_Single_Entry",
			"Extrae_OpenMP_Single_Exit");
		wrapRoutine (appImage, "_xlsmpBarrier_TPO",
		  "Extrae_OpenMP_Barrier_Entry",
			"Extrae_OpenMP_Barrier_Exit");

		if (locks)
		{
			wrapRoutine (appImage, "_xlsmpRelDefaultSLock",
			  "Extrae_OpenMP_Unnamed_Lock_Entry",
			  "Extrae_OpenMP_Unnamed_Unlock_Exit");
			wrapRoutine (appImage, "_xlsmpGetDefaultSLock",
			  "Extrae_OpenMP_Unnamed_Unlock_Entry",
			  "Extrae_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, "_xlsmpRelSLock",
			  "Extrae_OpenMP_Named_Lock_Entry",
			  "Extrae_OpenMP_Named_Unlock_Exit");
			wrapRoutine (appImage, "_xlsmpGetSLock",
			  "Extrae_OpenMP_Named_Unlock_Entry",
			  "Extrae_OpenMP_Named_Lock_Exit");
		}
	}

	if (at->get_OpenMP_rte() == ApplicationType::GNU_v42)
	{
		wrapRoutine (appImage, "GOMP_parallel_start",
		  "Extrae_OpenMP_ParRegion_Entry","");
		wrapRoutine (appImage, "GOMP_parallel_sections_start",
		  "Extrae_OpenMP_ParSections_Entry", "");
		wrapRoutine (appImage, "GOMP_parallel_end",
		  "", "Extrae_OpenMP_ParRegion_Exit");
		wrapRoutine (appImage, "GOMP_sections_start",
		  "Extrae_OpenMP_Section_Entry",
		  "Extrae_OpenMP_Section_Exit");
		wrapRoutine (appImage, "GOMP_sections_next",
		  "Extrae_OpenMP_Work_Entry",
		  "Extrae_OpenMP_Work_Exit");
		wrapRoutine (appImage, "GOMP_sections_end",
		  "Extrae_OpenMP_Join_Wait_Entry",
		  "Extrae_OpenMP_Join_Wait_Exit");
		wrapRoutine (appImage, "GOMP_sections_end_nowait",
		  "Extrae_OpenMP_Join_NoWait_Entry",
		  "Extrae_OpenMP_Join_NoWait_Exit");
		wrapRoutine (appImage, "GOMP_loop_end",
		  "Extrae_OpenMP_Join_Wait_Entry",
		  "Extrae_OpenMP_Join_Wait_Exit");
		wrapRoutine (appImage, "GOMP_loop_end_nowait",
		  "Extrae_OpenMP_Join_NoWait_Entry",
		  "Extrae_OpenMP_Join_NoWait_Exit");
		wrapRoutine (appImage, "GOMP_loop_static_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_loop_dynamic_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_loop_guided_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_loop_runtime_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_loop_ordered_static_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_loop_ordered_runtime_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_loop_ordered_dynamic_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_loop_ordered_guided_start",
		  "Extrae_OpenMP_DO_Entry",
		  "Extrae_OpenMP_DO_Exit");
		wrapRoutine (appImage, "GOMP_parallel_loop_static_start",
		  "Extrae_OpenMP_ParDO_Entry",
		  "Extrae_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, "GOMP_parallel_loop_dynamic_start",
		  "Extrae_OpenMP_ParDO_Entry",
		  "Extrae_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, "GOMP_parallel_loop_guided_start",
		  "Extrae_OpenMP_ParDO_Entry",
		  "Extrae_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, "GOMP_parallel_loop_runtime_start",
		  "Extrae_OpenMP_ParDO_Entry",
		  "Extrae_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, "GOMP_loop_static_next",
		  "Extrae_OpenMP_Work_Entry",
		  "Extrae_OpenMP_Work_Exit");
		wrapRoutine (appImage, "GOMP_loop_dynamic_next",
		  "Extrae_OpenMP_Work_Entry",
		  "Extrae_OpenMP_Work_Exit");
		wrapRoutine (appImage, "GOMP_loop_guided_next",
		  "Extrae_OpenMP_Work_Entry",
		  "Extrae_OpenMP_Work_Exit");
		wrapRoutine (appImage, "GOMP_loop_runtime_next",
		  "Extrae_OpenMP_Work_Entry",
		  "Extrae_OpenMP_Work_Exit");
		wrapRoutine (appImage, "GOMP_barrier",
		  "Extrae_OpenMP_Barrier_Entry",
		  "Extrae_OpenMP_Barrier_Exit");

		if (locks)
		{
			wrapRoutine (appImage, "GOMP_critical_name_start",
			  "Extrae_OpenMP_Named_Lock_Entry",
			  "Extrae_OpenMP_Named_Lock_Exit");
			wrapRoutine (appImage, "GOMP_critical_name_end",
			  "Extrae_OpenMP_Named_Unlock_Entry",
			  "Extrae_OpenMP_Named_Unlock_Exit");
			wrapRoutine (appImage, "GOMP_critical_start",
			  "Extrae_OpenMP_Unnamed_Lock_Entry",
			  "Extrae_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, "GOMP_critical_end",
			  "Extrae_OpenMP_Unnamed_Unlock_Entry",
			  "Extrae_OpenMP_Unnamed_Unlock_Exit");
			wrapRoutine (appImage, "GOMP_atomic_start",
			  "Extrae_OpenMP_Unnamed_Lock_Entry",
			  "Extrae_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, "GOMP_atomic_end",
			  "Extrae_OpenMP_Unnamed_Unlock_Entry",
			  "Extrae_OpenMP_Unnamed_Unlock_Exit");
		}
	}
}
