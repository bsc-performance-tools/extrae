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
Probe_GPI_connect_Entry(const gaspi_rank_t rank)
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_CONNECT_EV, EVT_BEGIN,
		    Extrae_get_trace_GPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GPI_RANK_EV, (int)rank);
	}
}

void
Probe_GPI_connect_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_CONNECT_EV, EVT_END,
		    Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_disconnect_Entry(const gaspi_rank_t rank)
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_DISCONNECT_EV, EVT_BEGIN,
		    Extrae_get_trace_GPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GPI_RANK_EV, (int)rank);
	}
}

void
Probe_GPI_disconnect_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_DISCONNECT_EV, EVT_END,
		    Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_group_create_Entry()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_GROUP_CREATE_EV, EVT_BEGIN,
		    Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_group_create_Exit(const gaspi_group_t *group)
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_GROUP_CREATE_EV, EVT_END,
		    Extrae_get_trace_GPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GPI_GROUP_EV, (int)group);
	}
}

void
Probe_GPI_group_add_Entry(const gaspi_group_t group, const gaspi_rank_t rank)
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_GROUP_ADD_EV, EVT_BEGIN,
		    Extrae_get_trace_GPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GPI_GROUP_EV, (int)group);
		TRACE_EVENT(LAST_READ_TIME, GPI_RANK_EV, (int)rank);
	}
}

void
Probe_GPI_group_add_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_GROUP_ADD_EV, EVT_END,
		    Extrae_get_trace_GPI_HWC());
	}
}

void
Probe_GPI_group_commit_Entry(const gaspi_group_t group,
    const gaspi_timeout_t timeout)
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_GROUP_COMMIT_EV, EVT_BEGIN,
		    Extrae_get_trace_GPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GPI_GROUP_EV, (int)group);
	}
}

void
Probe_GPI_group_commit_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_GROUP_COMMIT_EV, EVT_END,
		    Extrae_get_trace_GPI_HWC());
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
Probe_GPI_segment_create_Entry(const gaspi_segment_id_t segment_id,
    const gaspi_size_t size,
    const gaspi_group_t group)
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(
		    LAST_READ_TIME, GPI_SEGMENT_CREATE_EV, EVT_BEGIN,
		    Extrae_get_trace_GPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GPI_SEGMENT_ID_EV, (int)segment_id);
		TRACE_EVENT(LAST_READ_TIME, GPI_SIZE_EV, (int)size);
		TRACE_EVENT(LAST_READ_TIME, GPI_GROUP_EV, (int)group);
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

void
Probe_GPI_allreduce_Entry(const gaspi_number_t num,
    const gaspi_datatype_t datatyp, const gaspi_group_t group)
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(LAST_READ_TIME, GPI_ALLREDUCE_EV, EVT_BEGIN, Extrae_get_trace_GPI_HWC());
		TRACE_EVENT(LAST_READ_TIME, GPI_GROUP_EV, (int)group);
	}
}

void
Probe GPI_allreduce_Exit()
{
	if (mpitrace_on && Extrae_get_trace_GPI())
	{
		TRACE_EVENTANDCOUNTERS(TIME, GPI_ALLREDUCE_EV, EVT_END, Extrae_get_trace_GPI_HWC());
	}
}
