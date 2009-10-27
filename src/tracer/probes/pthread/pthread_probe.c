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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/probes/pthread/pthread_probe.c,v $
 | 
 | @last_commit: $Date: 2008/01/26 11:18:22 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: pthread_probe.c,v 1.3 2008/01/26 11:18:22 harald Exp $";

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "pthread_probe.h"

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

void Probe_pthread_Create_Entry (void *p)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADCREATE_EV, (UINT64) p, EMPTY);
}

void Probe_pthread_Create_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADCREATE_EV, EVT_END, EMPTY);
}

void Probe_pthread_Join_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADJOIN_EV, EVT_BEGIN, EMPTY);
}

void Probe_pthread_Join_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADJOIN_EV, EVT_END, EMPTY);
}

void Probe_pthread_Detach_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADDETACH_EV, EVT_BEGIN, EMPTY);
}

void Probe_pthread_Detach_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADDETACH_EV, EVT_END, EMPTY);
}
