/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/launcher/dyninst/ompSnippets.C,v $
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

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

#if 0
static void instrument_OMPRoutine (string routine, char *wrapper_begin,
	char *wrapper_end, BPatch_image *appImage, BPatch_process *appProcess)
{
	BPatch_function *func = getRoutine (routine, appImage);

	BPatch_Vector<BPatch_point *> *entry_point = func->findPoint(BPatch_entry);
	BPatch_Vector<BPatch_point *> *exit_point = func->findPoint(BPatch_exit);

	if (!entry_point || entry_point->size() == 0)
	{
		fprintf (stderr, "** FAILED, unable to find entry point to \"%s\"\n", routine.c_str());
		return;
	}
	if (!exit_point || exit_point->size() == 0)
	{
		fprintf (stderr, "** FAILED, unable to find exit point to \"%s\"\n", routine.c_str());
		return;
	}

	if (wrapper_begin != NULL)
	{
		BPatch_Vector<BPatch_function *> snippet_begin;
		if (appImage->findFunction (wrapper_begin, snippet_begin) == NULL || snippet_begin.size() == 0 || snippet_begin[0] == NULL)
		{
			fprintf (stderr, "Failed when finding \"%s\"\n", wrapper_begin);
			return;
		}
		BPatch_Vector<BPatch_snippet *> args_entry;
		BPatch_funcCallExpr callExpr_entry (*(snippet_begin[0]), args_entry);
		appProcess->insertSnippet (callExpr_entry, *entry_point);
	}

	if (wrapper_end != NULL)
	{
		BPatch_Vector<BPatch_function *> snippet_end;
		if (appImage->findFunction (wrapper_end, snippet_end) == NULL || snippet_end.size() == 0 || snippet_end[0] == NULL)
		{
			fprintf (stderr, "Failed when finding \"%s\"\n", wrapper_end);
			return;
		}
		BPatch_Vector<BPatch_snippet *> args_exit;
		BPatch_funcCallExpr callExpr_exit (*(snippet_end[0]), args_exit);
		appProcess->insertSnippet (callExpr_exit, *exit_point);
	}
}

void InsertOMPSnippets (bool locks, BPatch_image *appImage, BPatch_process *appProcess)
{
	instrument_OMPRoutine ("__kmpc_fork_call",
		"Probe_OpenMP_ParRegion_Entry", "Probe_OpenMP_ParRegion_Exit",
		appImage, appProcess);
	instrument_OMPRoutine ("__kmpc_barrier",
		"Probe_OpenMP_Barrier_Entry", "Probe_OpenMP_Barrier_Exit",
		appImage, appProcess);
	instrument_OMPRoutine ("__kmpc_invoke_task_func",
		"Probe_OpenMP_ParDO_Entry", "Probe_OpenMP_ParDO_Exit",
		appImage, appProcess);
	if (locks)
	{
		instrument_OMPRoutine ("__kmpc_set_lock",
			"Probe_OpenMP_Unnamed_Lock_Entry", "Probe_OpenMP_Unnamed_Lock_Exit",
			appImage, appProcess);
		instrument_OMPRoutine ("__kmpc_unset_lock",
			"Probe_OpenMP_Unnamed_Unlock_Entry", "Probe_OpenMP_Unnamed_Unlock_Exit",
			appImage, appProcess);
	}
}
#endif

void InstrumentOMPruntime (bool locks, ApplicationType *at, BPatch_image *appImage,
	BPatch_process *appProcess)
{
	if (at->get_OpenMP_rte() == ApplicationType::Intel_v81 ||
	    at->get_OpenMP_rte() == ApplicationType::Intel_v91)
	{
		wrapRoutine (appImage, appProcess, "__kmpc_fork_call",
		  "Probe_OpenMP_ParRegion_Entry", "Probe_OpenMP_ParRegion_Exit");
		wrapRoutine (appImage, appProcess, "__kmpc_barrier",
		  "Probe_OpenMP_Barrier_Entry", "Probe_OpenMP_Barrier_Exit");
		wrapRoutine (appImage, appProcess, "__kmpc_invoke_task_func",
		  "Probe_OpenMP_ParDO_Entry", "Probe_OpenMP_ParDO_Exit");
		if (locks)
		{
			wrapRoutine (appImage, appProcess, "__kmpc_set_lock",
			  "Probe_OpenMP_Unnamed_Lock_Entry", "Probe_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, appProcess, "__kmpc_unset_lock",
			  "Probe_OpenMP_Unnamed_Unlock_Entry", "Probe_OpenMP_Unnamed_Unlock_Exit");
			wrapRoutine (appImage, appProcess, "__kmpc_critical",
			  "Probe_OpenMP_Named_Lock_Entry", "Probe_OpenMP_Named_Lock_Exit");
			wrapRoutine (appImage, appProcess, "__kmpc_end_critical",
			  "Probe_OpenMP_Named_Unlock_Entry", "Probe_OpenMP_Named_Unlock_Exit");
		}
	}

	if (at->get_OpenMP_rte() == ApplicationType::IBM_v16)
	{
		wrapRoutine (appImage, appProcess, "_xlsmpParallelDoSetup_TPO",
		  "Probe_OpenMP_ParDO_Entry", "Probe_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, appProcess, "_xlsmpWSDoSetup_TPO",
		  "Probe_OpenMP_DO_Entry", "Probe_OpenMP_DO_Exit");
		wrapRoutine (appImage, appProcess, "_xlsmpParRegionSetup_TPO",
		  "Probe_OpenMP_ParRegion_Entry", "Probe_OpenMP_ParRegion_Exit");
		wrapRoutine (appImage, appProcess, "_xlsmpWSSectSetup_TPO",
		  "Probe_OpenMP_Section_Entry", "Probe_OpenMP_Section_Exit");
		wrapRoutine (appImage, appProcess, "_xlsmpSingleSetup_TPO",
		  "Probe_OpenMP_Single_Entry", "Probe_OpenMP_Single_Exit");
		wrapRoutine (appImage, appProcess, "_xlsmpBarrier_TPO",
		  "Probe_OpenMP_Barrier_Entry", "Probe_OpenMP_Barrier_Exit");

		if (locks)
		{
			wrapRoutine (appImage, appProcess, "_xlsmpRelDefaultSLock",
			  "Probe_OpenMP_Unnamed_Lock_Entry", "Probe_OpenMP_Unnamed_Unlock_Exit");
			wrapRoutine (appImage, appProcess, "_xlsmpGetDefaultSLock",
			  "Probe_OpenMP_Unnamed_Unlock_Entry", "Probe_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, appProcess, "_xlsmpRelSLock",
			  "Probe_OpenMP_Named_Lock_Entry", "Probe_OpenMP_Named_Unlock_Exit");
			wrapRoutine (appImage, appProcess, "_xlsmpGetSLock",
			  "Probe_OpenMP_Named_Unlock_Entry", "Probe_OpenMP_Named_Lock_Exit");
		}
	}

	if (at->get_OpenMP_rte() == ApplicationType::GNU_v42)
	{
		wrapRoutine (appImage, appProcess, "GOMP_parallel_start",
			"Probe_OpenMP_ParRegion_Entry","");
		wrapRoutine (appImage, appProcess, "GOMP_parallel_sections_start",
		  "Probe_OpenMP_ParSections_Entry", "");
		wrapRoutine (appImage, appProcess, "GOMP_parallel_end",
			"", "Probe_OpenMP_ParRegion_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_sections_start",
		  "Probe_OpenMP_Section_Entry", "Probe_OpenMP_Section_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_sections_next",
		  "Probe_OpenMP_Work_Entry", "Probe_OpenMP_Work_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_sections_end",
		  "Probe_OpenMP_Join_Wait_Entry", "Probe_OpenMP_Join_Wait_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_sections_end_nowait",
		  "Probe_OpenMP_Join_NoWait_Entry", "Probe_OpenMP_Join_NoWait_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_loop_end",
		  "Probe_OpenMP_Join_Wait_Entry", "Probe_OpenMP_Join_Wait_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_loop_end_nowait",
		  "Probe_OpenMP_Join_NoWait_Entry", "Probe_OpenMP_Join_NoWait_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_loop_static_start",
		  "Probe_OpenMP_DO_Entry", "Probe_OpenMP_DO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_loop_dynamic_start",
		  "Probe_OpenMP_DO_Entry", "Probe_OpenMP_DO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_loop_guided_start",
		  "Probe_OpenMP_DO_Entry", "Probe_OpenMP_DO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_loop_runtime_start",
		  "Probe_OpenMP_DO_Entry", "Probe_OpenMP_DO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_parallel_loop_static_start",
		  "Probe_OpenMP_ParDO_Entry", "Probe_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_parallel_loop_dynamic_start",
		  "Probe_OpenMP_ParDO_Entry", "Probe_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_parallel_loop_guided_start",
		  "Probe_OpenMP_ParDO_Entry", "Probe_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_parallel_loop_runtime_start",
		  "Probe_OpenMP_ParDO_Entry", "Probe_OpenMP_ParDO_Exit");
		wrapRoutine (appImage, appProcess, "GOMP_loop_static_next",
		  "Probe_OpenMP_Work_Entry", "Probe_OpenMP_Work_Entry");
		wrapRoutine (appImage, appProcess, "GOMP_loop_dynamic_next",
		  "Probe_OpenMP_Work_Entry", "Probe_OpenMP_Work_Entry");
		wrapRoutine (appImage, appProcess, "GOMP_loop_guided_next",
		  "Probe_OpenMP_Work_Entry", "Probe_OpenMP_Work_Entry");
		wrapRoutine (appImage, appProcess, "GOMP_loop_runtime_next",
		  "Probe_OpenMP_Work_Entry", "Probe_OpenMP_Work_Entry");
		wrapRoutine (appImage, appProcess, "GOMP_barrier",
		  "Probe_OpenMP_Barrier_Entry", "Probe_OpenMP_Barrier_Exit");

		if (locks)
		{
			wrapRoutine (appImage, appProcess, "GOMP_critical_name_start",
			  "Probe_OpenMP_Named_Lock_Entry", "Probe_OpenMP_Named_Lock_Exit");
			wrapRoutine (appImage, appProcess, "GOMP_critical_name_end",
			  "Probe_OpenMP_Named_Unlock_Entry", "Probe_OpenMP_Named_Unlock_Exit");
			wrapRoutine (appImage, appProcess, "GOMP_critical_start",
			  "Probe_OpenMP_Unnamed_Lock_Entry", "Probe_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, appProcess, "GOMP_critical_end",
			  "Probe_OpenMP_Unnamed_Unlock_Entry", "Probe_OpenMP_Unnamed_Unlock_Exit");
			wrapRoutine (appImage, appProcess, "GOMP_atomic_start",
			  "Probe_OpenMP_Unnamed_Lock_Entry", "Probe_OpenMP_Unnamed_Lock_Exit");
			wrapRoutine (appImage, appProcess, "GOMP_atomic_end",
			  "Probe_OpenMP_Unnamed_Unlock_Entry", "Probe_OpenMP_Unnamed_Unlock_Exit");
		}
	}

}
