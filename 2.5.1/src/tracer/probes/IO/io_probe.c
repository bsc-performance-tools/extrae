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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.4/src/tracer/wrappers/OMP/omp_wrapper.c $
 | @last_commit: $Date: 2013-09-06 14:39:32 +0200 (Fri, 06 Sep 2013) $
 | @version:     $Revision: 2098 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: omp_wrapper.c 2098 2013-09-06 12:39:32Z harald $";

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "io_probe.h"

static int trace_io = FALSE;

void Extrae_set_trace_io (int b)
{ trace_io = b; }

int Extrae_get_trace_io (void)
{ return trace_io; }

void Probe_IO_write_Entry (int f, size_t s)
{
	if (mpitrace_on && trace_io)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, WRITE_EV, EVT_BEGIN, f);
		TRACE_MISCEVENT(LAST_READ_TIME, WRITE_EV, EVT_BEGIN+1, s);
	}
}

void Probe_IO_write_Exit (void)
{
	if (mpitrace_on && trace_io)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, WRITE_EV, EVT_END, EMPTY);
	}
}

void Probe_IO_read_Entry (int f, size_t s)
{
	if (mpitrace_on && trace_io)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, READ_EV, EVT_BEGIN, f);
		TRACE_MISCEVENT(LAST_READ_TIME, READ_EV, EVT_BEGIN+1, s);
	}
}

void Probe_IO_read_Exit (void)
{
	if (mpitrace_on && trace_io)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, READ_EV, EVT_END, EMPTY);
	}
}
