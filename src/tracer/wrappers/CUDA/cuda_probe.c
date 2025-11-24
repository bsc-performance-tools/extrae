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
#include "cuda_probe.h"

int trace_cuda = TRUE;

void Extrae_set_trace_CUDA (int b)
{ trace_cuda = b; }

int Extrae_get_trace_CUDA (void)
{ return trace_cuda; }

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

void Probe_Cuda_ConfigureCall_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDACONFIGCALL_VAL, EVT_BEGIN);
}

void Probe_Cuda_ConfigureCall_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDACONFIGCALL_VAL, EVT_END);
}

void Probe_Cuda_Launch_Entry (UINT64 p1)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDALAUNCH_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_KERNEL_INST_EV, p1);
	}
}

void Probe_Cuda_Launch_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDALAUNCH_VAL, EVT_END);
		TRACE_EVENT(LAST_READ_TIME, CUDA_KERNEL_INST_EV, EVT_END);
	}
}

void Probe_Cuda_Malloc_Entry(unsigned int event, UINT64 ptr, size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, event, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_PTR_EV, ptr);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Cuda_Malloc_Exit(unsigned int event)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, event, EVT_END);
	}
}

void Probe_Cuda_Free_Entry(unsigned int event, UINT64 devPtr)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, event, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_PTR_EV, devPtr);
	}
}

void Probe_Cuda_Free_Exit(unsigned int event)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, event, EVT_END);
	}
}

void Probe_Cuda_HostAlloc_Entry(UINT64 ptr, size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDAHOSTALLOC_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_PTR_EV, ptr);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Cuda_HostAlloc_Exit()
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDAHOSTALLOC_VAL, EVT_END);
	}
}

void Probe_Cuda_Memcpy_Entry (size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDAMEMCPY_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Cuda_Memcpy_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDAMEMCPY_VAL, EVT_END); 
	}
}

void Probe_Cuda_MemcpyAsync_Entry (size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDAMEMCPYASYNC_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_SIZE_EV, size);
	}
}

void Probe_Cuda_MemcpyAsync_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDAMEMCPYASYNC_VAL, EVT_END); 
	}
}

void Probe_Cuda_Memset_Entry(UINT64 devPtr, size_t count)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDAMEMSET_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_PTR_EV, devPtr);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_SIZE_EV, count);
	}
}

void Probe_Cuda_Memset_Exit()
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDAMEMSET_VAL, EVT_END);
	}
}

void Probe_Cuda_MemsetAsync_Entry(UINT64 devPtr, size_t count)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDAMEMSETASYNC_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_PTR_EV, devPtr);
		TRACE_EVENT(LAST_READ_TIME, CUDA_DYNAMIC_MEM_SIZE_EV, count);
	}
}

void Probe_Cuda_MemsetAsync_Exit()
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDAMEMSETASYNC_VAL, EVT_END);
	}
}

void Probe_Cuda_ThreadBarrier_Entry(void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDATHREADBARRIER_VAL, EVT_BEGIN);
}

void Probe_Cuda_ThreadBarrier_Exit(void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDATHREADBARRIER_VAL, EVT_END); 
}

void Probe_Cuda_StreamBarrier_Entry (unsigned threadid)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDASTREAMBARRIER_VAL, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, CUDASTREAMBARRIER_THID_EV, threadid+1);
	}
}

void Probe_Cuda_StreamBarrier_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDASTREAMBARRIER_VAL, EVT_END); 
	}
}


void Probe_Cuda_DeviceReset_Enter (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDADEVICERESET_VAL, EVT_BEGIN); 
}

void Probe_Cuda_DeviceReset_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDADEVICERESET_VAL, EVT_END); 
}

void Probe_Cuda_ThreadExit_Enter (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDATHREADEXIT_VAL, EVT_BEGIN); 
}

void Probe_Cuda_ThreadExit_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDATHREADEXIT_VAL, EVT_END); 
}

void Probe_Cuda_StreamDestroy_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDASTREAMDESTROY_VAL, EVT_BEGIN);
}

void Probe_Cuda_StreamDestroy_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDASTREAMDESTROY_VAL, EVT_END);
}

void Probe_Cuda_StreamCreate_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDASTREAMCREATE_VAL, EVT_BEGIN);
}

void Probe_Cuda_StreamCreate_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDASTREAMCREATE_VAL, EVT_END);
}

void Probe_Cuda_EventRecord_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDAEVENTRECORD_VAL, EVT_BEGIN);
}

void Probe_Cuda_EventRecord_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDAEVENTRECORD_VAL, EVT_END);
}


void Probe_Cuda_EventSynchronize_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDAEVENTSYNCHRONIZE_VAL, EVT_BEGIN);
}

void Probe_Cuda_EventSynchronize_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDAEVENTSYNCHRONIZE_VAL, EVT_END);
}

void Probe_Cuda_StreamWaitEvent_Enter (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACALL_EV, CUDASTREAMWAITEVENT_VAL, EVT_BEGIN);
}

void Probe_Cuda_StreamWaitEvent_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
	    TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACALL_EV, CUDASTREAMWAITEVENT_VAL, EVT_END);
}