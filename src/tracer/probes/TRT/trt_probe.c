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
#include "trt_probe.h"

#if 1
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

void Probe_threadSpawn_Entry (void *p)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADCREATE_EV, (UINT64) p, EMPTY);
}

void Probe_threadSpawn_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADCREATE_EV, EVT_END, EMPTY);
}

void Probe_threadRead_Entry (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADJOIN_EV, EVT_BEGIN, EMPTY);
}

void Probe_threadRead_Exit (void)
{
	DEBUG
	if (mpitrace_on)
		TRACE_OMPEVENTANDCOUNTERS(TIME, PTHREADJOIN_EV, EVT_END, EMPTY);
}
