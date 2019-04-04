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

#include "wrapper.h"

#include "gpi_probe.h"

static int trace_gpi = TRUE;
static int trace_gpi_hwc = TRUE;

void
Extrae_set_trace_GPI(int trace)
{
	trace_gpi = trace;
}

int
Extrae_get_trace_GPI()
{
	return trace_gpi;
}

void
Extrae_set_trace_GPI_HWC(int trace)
{
	trace_gpi_hwc = trace;
}

int
Extrae_get_trace_GPI_HWC()
{
	return trace_gpi_hwc;
}

void
Probe_GPI_init_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_INIT_EV, EVT_BEGIN, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_init_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_INIT_EV, EVT_END, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_term_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_TERM_EV, EVT_BEGIN, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_term_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_TERM_EV, EVT_END, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_barrier_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_BARRIER_EV, EVT_BEGIN, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_barrier_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_BARRIER_EV, EVT_END, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_segment_create_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_SEGMENT_CREATE_EV, EVT_BEGIN, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_segment_create_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_SEGMENT_CREATE_EV, EVT_END, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_write_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_WRITE_EV, EVT_BEGIN, Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_write_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_WRITE_EV, EVT_END, Extrae_get_trace_GPI_HWC());
	}
}
