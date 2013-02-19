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

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "trt_probe.h"

#if 0
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
