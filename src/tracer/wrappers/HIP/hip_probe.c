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
#include "hip_probe.h"

int trace_hip = TRUE;

void Extrae_set_trace_HIP(int b)
{ trace_hip = b; }

int Extrae_get_trace_HIP(void)
{ return trace_hip; }

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

void Probe_Hip_ConfigureCall_Entry(void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPCONFIGCALL_VAL, EVT_BEGIN);
}

void Probe_Hip_ConfigureCall_Exit(void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPCONFIGCALL_VAL, EVT_END);
}

void Probe_Hip_Launch_Entry(UINT64 p1)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPLAUNCH_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIP_KERNEL_INST_EV, p1);
	}
}

void Probe_Hip_Launch_Exit(void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPLAUNCH_VAL, EVT_END);
		TRACE_EVENT(LAST_READ_TIME, HIP_KERNEL_INST_EV, EVT_END);
	}
}

void Probe_Hip_Malloc_Entry(unsigned int event, UINT64 ptr, size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, event, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_PTR_EV, ptr);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Hip_Malloc_Exit(unsigned int event)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, event, EVT_END);
	}
}

void Probe_Hip_Free_Entry(unsigned int event, UINT64 devPtr)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, event, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_PTR_EV, devPtr);
	}
}

void Probe_Hip_Free_Exit(unsigned int event)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, event, EVT_END);
	}
}

void Probe_Hip_HostAlloc_Entry(UINT64 ptr, size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPHOSTALLOC_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_PTR_EV, ptr);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Hip_HostAlloc_Exit()
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPHOSTALLOC_VAL, EVT_END);
	}
}

void Probe_Hip_Memcpy_Entry (size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPMEMCPY_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Hip_Memcpy_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPMEMCPY_VAL, EVT_END); 
	}
}

void Probe_Hip_MemcpyAsync_Entry (size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPMEMCPYASYNC_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Hip_MemcpyAsync_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPMEMCPYASYNC_VAL, EVT_END); 
	}
}

void Probe_Hip_Memset_Entry(UINT64 devPtr, size_t count)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPMEMSET_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_PTR_EV, devPtr);
		TRACE_EVENT(LAST_READ_TIME, HIP_DYNAMIC_MEM_SIZE_EV, count);
	}
}

void Probe_Hip_Memset_Exit()
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPMEMSET_VAL, EVT_END);
	}
}

void Probe_Hip_ThreadBarrier_Entry(void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPTHREADBARRIER_VAL, EVT_BEGIN);
}

void Probe_Hip_ThreadBarrier_Exit(void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPTHREADBARRIER_VAL, EVT_END); 
}

void Probe_Hip_StreamBarrier_Entry (unsigned threadid)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPSTREAMBARRIER_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, HIPSTREAMBARRIER_THID_EV, threadid+1);
	}
}

void Probe_Hip_StreamBarrier_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPSTREAMBARRIER_VAL, EVT_END); 
	}
}


void Probe_Hip_DeviceReset_Enter (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPDEVICERESET_VAL, EVT_BEGIN); 
}

void Probe_Hip_DeviceReset_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPDEVICERESET_VAL, EVT_END); 
}

void Probe_Hip_ThreadExit_Enter (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPTHREADEXIT_VAL, EVT_BEGIN); 
}

void Probe_Hip_ThreadExit_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
		TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPTHREADEXIT_VAL, EVT_END); 
}

void Probe_Hip_StreamDestroy_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPSTREAMDESTROY_VAL, EVT_BEGIN);
}

void Probe_Hip_StreamDestroy_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPSTREAMDESTROY_VAL, EVT_END);
}

void Probe_Hip_StreamCreate_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPSTREAMCREATE_VAL, EVT_BEGIN);
}

void Probe_Hip_StreamCreate_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPSTREAMCREATE_VAL, EVT_END);
}

void Probe_Hip_EventRecord_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPEVENTRECORD_VAL, EVT_BEGIN);
}

void Probe_Hip_EventRecord_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPEVENTRECORD_VAL, EVT_END);
}


void Probe_Hip_EventSynchronize_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, HIPCALL_EV, HIPEVENTSYNCHRONIZE_VAL, EVT_BEGIN);
}

void Probe_Hip_EventSynchronize_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_HIP())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, HIPCALL_EV, HIPEVENTSYNCHRONIZE_VAL, EVT_END);
}
