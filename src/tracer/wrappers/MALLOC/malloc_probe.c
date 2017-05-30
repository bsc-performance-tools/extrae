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

void Probe_posix_memalign_Entry(size_t size)
{
        if (mpitrace_on && trace_malloc)
        {
                TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, POSIX_MEMALIGN_EV, EVT_BEGIN, size);
        }
}

void Probe_posix_memalign_Exit(void *ptr)
{
        if (mpitrace_on && trace_malloc)
        {
                TRACE_MISCEVENTANDCOUNTERS(TIME, POSIX_MEMALIGN_EV, EVT_END, ptr);
        }
}

void Probe_memkind_malloc_Entry(int kind, size_t size)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, MEMKIND_MALLOC_EV, EVT_BEGIN, size);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, kind, EMPTY);
	}
}

void Probe_memkind_malloc_Exit(void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, MEMKIND_MALLOC_EV, EVT_END, (UINT64) ptr);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, EMPTY, EMPTY);
	}
}

void Probe_memkind_calloc_Entry(int kind, size_t num, size_t size)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, MEMKIND_CALLOC_EV, EVT_BEGIN, num*size);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, kind, EMPTY);
	}
}

void Probe_memkind_calloc_Exit(void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, MEMKIND_CALLOC_EV, EVT_END, (UINT64) ptr);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, EMPTY, EMPTY);
	}
}

void Probe_memkind_realloc_Entry(int kind, void *ptr, size_t size)
{
	if (mpitrace_on && trace_malloc)
	{
		/* Split ptr & size in two events. There's no need to read counters for the second event */
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, MEMKIND_REALLOC_EV, EVT_BEGIN, (UINT64) ptr);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_REALLOC_EV, EVT_BEGIN+1, size);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, kind, EMPTY);
	}
}

void Probe_memkind_realloc_Exit(void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, MEMKIND_REALLOC_EV, EVT_END, (UINT64) ptr);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, EMPTY, EMPTY);
	}
}

void Probe_memkind_posix_memalign_Entry(int kind, size_t size)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, MEMKIND_POSIX_MEMALIGN_EV, EVT_BEGIN, size);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, kind, EMPTY);
	}
}

void Probe_memkind_posix_memalign_Exit(void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, MEMKIND_POSIX_MEMALIGN_EV, EVT_END, ptr);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, EMPTY, EMPTY);
	}
}

void Probe_memkind_free_Entry(int kind, void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, MEMKIND_FREE_EV, EVT_BEGIN, (UINT64) ptr);
                TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, kind, EMPTY);
	}
}

void Probe_memkind_free_Exit()
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, MEMKIND_FREE_EV, EVT_END, EMPTY);
		TRACE_MISCEVENT(LAST_READ_TIME, MEMKIND_PARTITION_EV, EMPTY, EMPTY);
	}
}

void Probe_kmpc_malloc_Entry(size_t size)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, KMPC_MALLOC_EV, EVT_BEGIN, size);
	}
}

void Probe_kmpc_malloc_Exit(void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, KMPC_MALLOC_EV, EVT_END, (UINT64) ptr);
	}
}

void
Probe_kmpc_aligned_malloc_Entry(size_t size, size_t alignment)
{
	UNREFERENCED_PARAMETER(alignment);

	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, KMPC_ALIGNED_MALLOC_EV, EVT_BEGIN, size);
	}
}

void Probe_kmpc_aligned_malloc_Exit(void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, KMPC_ALIGNED_MALLOC_EV, EVT_END, (UINT64) ptr);
	}
}

void Probe_kmpc_calloc_Entry(size_t nelem, size_t elsize)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, KMPC_CALLOC_EV, EVT_BEGIN, nelem*elsize);
	}
}

void Probe_kmpc_calloc_Exit(void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, KMPC_CALLOC_EV, EVT_END, (UINT64) ptr);
	}
}

void Probe_kmpc_realloc_Entry (void *ptr, size_t size)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, KMPC_REALLOC_EV, EVT_BEGIN, (UINT64) ptr);
		TRACE_MISCEVENT(LAST_READ_TIME, KMPC_REALLOC_EV, EVT_BEGIN+1, size);
	}
}

void Probe_kmpc_realloc_Exit (void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, KMPC_REALLOC_EV, EVT_END, (UINT64) ptr);
	}
}

void Probe_kmpc_free_Entry (void *ptr)
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, KMPC_FREE_EV, EVT_BEGIN, (UINT64) ptr);
	}
}

void Probe_kmpc_free_Exit ()
{
	if (mpitrace_on && trace_malloc)
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, KMPC_FREE_EV, EVT_END, EMPTY);
	}
}
