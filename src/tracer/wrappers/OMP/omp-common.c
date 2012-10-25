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

#include "omp_probe.h"
#include "omp-common.h"
#include "wrapper.h"

void Extrae_OpenMP_Join_NoWait_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Join_NoWait_Entry ();
}

void Extrae_OpenMP_Join_NoWait_Exit (void)
{
	Probe_OpenMP_Join_NoWait_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Join_Wait_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Join_Wait_Entry ();
}

void Extrae_OpenMP_Join_Wait_Exit (void)
{
	Probe_OpenMP_Join_Wait_Exit ();	
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_UF_Entry (UINT64 uf)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_UF_Entry (uf);
}

void Extrae_OpenMP_UF_Exit (void)
{
	Probe_OpenMP_UF_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Work_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Work_Entry ();
}

void Extrae_OpenMP_Work_Exit (void)
{
	Probe_OpenMP_Work_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_DO_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_DO_Entry ();
}

void Extrae_OpenMP_DO_Exit (void)
{
	Probe_OpenMP_DO_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Sections_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Sections_Entry ();
}

void Extrae_OpenMP_Sections_Exit (void)
{
	Probe_OpenMP_Sections_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_ParRegion_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_ParRegion_Entry ();
}

void Extrae_OpenMP_ParRegion_Exit (void)
{
	Probe_OpenMP_ParRegion_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_ParDO_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_ParDO_Entry ();
}

void Extrae_OpenMP_ParDO_Exit (void)
{
	Probe_OpenMP_ParDO_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_ParSections_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_ParSections_Entry ();
}

void Extrae_OpenMP_ParSections_Exit (void)
{
	Probe_OpenMP_ParSections_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Barrier_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Barrier_Entry ();
}

void Extrae_OpenMP_Barrier_Exit (void)
{
	Probe_OpenMP_Barrier_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Single_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Single_Entry ();
}

void Extrae_OpenMP_Single_Exit (void)
{
	Probe_OpenMP_Single_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Section_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Section_Entry ();
}

void Extrae_OpenMP_Section_Exit (void)
{
	Probe_OpenMP_Sections_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Named_Lock_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Named_Lock_Entry ();
}

void Extrae_OpenMP_Named_Lock_Exit (void)
{
	Probe_OpenMP_Named_Lock_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Named_Unlock_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Named_Unlock_Entry ();
}

void Extrae_OpenMP_Named_Unlock_Exit (void)
{
	Probe_OpenMP_Named_Unlock_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Unnamed_Lock_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Unnamed_Lock_Entry ();
}

void Extrae_OpenMP_Unnamed_Lock_Exit (void)
{
	Probe_OpenMP_Unnamed_Lock_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Unnamed_Unlock_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Unnamed_Unlock_Entry ();
}

void Extrae_OpenMP_Unnamed_Unlock_Exit (void)
{
	Probe_OpenMP_Unnamed_Unlock_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_GetNumThreads_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_GetNumThreads_Entry ();
}

void Extrae_OpenMP_GetNumThreads_Exit (void)
{
	Probe_OpenMP_GetNumThreads_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_SetNumThreads_Entry (int p1)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_SetNumThreads_Entry (p1);
}

void Extrae_OpenMP_SetNumThreads_Exit (void)
{
	Probe_OpenMP_SetNumThreads_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Task_Entry (UINT64 uf)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Task_Entry (uf);
}

void Extrae_OpenMP_Task_Exit (void)
{
	Probe_OpenMP_Task_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_TaskUF_Entry (UINT64 uf)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_TaskUF_Entry (uf);
}

void Extrae_OpenMP_TaskUF_Exit (void)
{
	Probe_OpenMP_TaskUF_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_OpenMP_Taskwait_Entry (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_OpenMP_Taskwait_Entry ();
}

void Extrae_OpenMP_Taskwait_Exit (void)
{
	Probe_OpenMP_Taskwait_Exit ();
	Backend_Leave_Instrumentation ();
}

