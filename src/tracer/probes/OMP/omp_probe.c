/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
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
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "omp_probe.h"

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

static int TraceOMPLocks = FALSE;

void setTrace_OMPLocks (int value)
{
	TraceOMPLocks = value;
}

void Probe_OpenMP_Join_NoWait_Entry (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, JOIN_EV, JOIN_NOWAIT_VAL, EMPTY);
}

void Probe_OpenMP_Join_NoWait_Exit (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, JOIN_EV, EVT_END, EMPTY);
}

void Probe_OpenMP_Join_Wait_Entry (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, JOIN_EV, JOIN_WAIT_VAL, EMPTY);
}

void Probe_OpenMP_Join_Wait_Exit (void)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, JOIN_EV, EVT_END, EMPTY);
}

void Probe_OpenMP_UF (UINT64 uf)
{
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, OMPFUNC_EV, uf, EMPTY);
}

void Probe_OpenMP_Work_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WORK_EV, EVT_BEGIN, EMPTY);
}

void Probe_OpenMP_Work_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WORK_EV, EVT_END, EMPTY);
}

void Probe_OpenMP_DO_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_DO_VAL, EMPTY);
}

void Probe_OpenMP_DO_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_END_VAL, EMPTY); 
}

void Probe_OpenMP_Sections_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_SEC_VAL, EMPTY);
}

void Probe_OpenMP_Sections_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_END_VAL, EMPTY); 
}

void Probe_OpenMP_ParRegion_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS (TIME, PAR_EV, PAR_REG_VAL, EMPTY);
}

void Probe_OpenMP_ParRegion_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_END_VAL, EMPTY);
}

void Probe_OpenMP_ParDO_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_WSH_VAL, EMPTY);
}

void Probe_OpenMP_ParDO_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_END_VAL, EMPTY);
}

void Probe_OpenMP_ParSections_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_SEC_VAL, EMPTY);
}

void Probe_OpenMP_ParSections_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PAR_EV, PAR_END_VAL, EMPTY);
}

void Probe_OpenMP_Barrier_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, BARRIEROMP_EV, EVT_BEGIN, EMPTY);
}

void Probe_OpenMP_Barrier_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, BARRIEROMP_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_Single_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_SINGLE_VAL, EMPTY);
}

void Probe_OpenMP_Single_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_Section_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, WSH_SEC_VAL, EMPTY);
}

void Probe_OpenMP_Section_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, WSH_EV, EVT_END, EMPTY); 
}

void Probe_OpenMP_Named_Lock_Entry (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, NAMEDCRIT_EV, LOCK_VAL, EMPTY);
}

void Probe_OpenMP_Named_Lock_Exit (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, NAMEDCRIT_EV, LOCKED_VAL, EMPTY);
}

void Probe_OpenMP_Named_Unlock_Entry (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, NAMEDCRIT_EV, UNLOCK_VAL, EMPTY);
}

void Probe_OpenMP_Named_Unlock_Exit (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, NAMEDCRIT_EV, UNLOCKED_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Lock_Entry (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, UNNAMEDCRIT_EV, LOCK_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Lock_Exit (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, UNNAMEDCRIT_EV, LOCK_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Unlock_Entry (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, UNNAMEDCRIT_EV, UNLOCK_VAL, EMPTY);
}

void Probe_OpenMP_Unnamed_Unlock_Exit (void)
{
	DEBUG
	if (TraceOMPLocks && mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, UNNAMEDCRIT_EV, UNLOCKED_VAL, EMPTY);
}
