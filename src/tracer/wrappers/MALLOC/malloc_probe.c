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

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "malloc_probe.h"

static int trace_malloc = FALSE;
static int trace_malloc_allocate = TRUE;
static int trace_malloc_free     = FALSE;
static unsigned long long trace_malloc_allocate_threshold = 1024;

void Extrae_set_trace_malloc_allocate (int b)
{ trace_malloc_allocate = b; }

int Extrae_get_trace_malloc_allocate (void)
{ return trace_malloc_allocate; }

void Extrae_set_trace_malloc_allocate_threshold (unsigned long long t)
{ trace_malloc_allocate_threshold = t; }

unsigned long long Extrae_get_trace_malloc_allocate_threshold (void)
{ return trace_malloc_allocate_threshold; }

void Extrae_set_trace_malloc_free (int b)
{ trace_malloc_free = b; }

int Extrae_get_trace_malloc_free (void)
{ return trace_malloc_free; }

void Extrae_set_trace_malloc (int b)
{ trace_malloc = b; }

int Extrae_get_trace_malloc (void)
{ return trace_malloc; }

void Probe_Malloc_Entry (size_t s)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, MALLOC_EV, EVT_BEGIN, s);
	}
}

void Probe_Malloc_Exit (void *p)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, MALLOC_EV, EVT_END, (UINT64) p);
	}
}

void Probe_Free_Entry (void *p)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, FREE_EV, EVT_BEGIN, (UINT64) p);
	}
}

void Probe_Free_Exit (void)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, FREE_EV, EVT_END, EMPTY);
	}
}

void Probe_Calloc_Entry (size_t s1, size_t s2)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CALLOC_EV, EVT_BEGIN, s1*s2);
	}
}

void Probe_Calloc_Exit (void *p)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CALLOC_EV, EVT_END, (UINT64) p);
	}
}

void Probe_Realloc_Entry (void *p, size_t s)
{
	if (mpitrace_on && trace_malloc)
	{
		/* Split p & s in two events. There's no need to read counters for the
		   second event */
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, REALLOC_EV, EVT_BEGIN, (UINT64) p);
		TRACE_MISCEVENT(LAST_READ_TIME, REALLOC_EV, EVT_BEGIN+1, s);
	}
}

void Probe_Realloc_Exit (void *p)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, REALLOC_EV, EVT_END, (UINT64) p);
	}
}
