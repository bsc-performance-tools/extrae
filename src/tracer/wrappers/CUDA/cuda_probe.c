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

void Probe_Cuda_Launch_Entry (UINT64 p1)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDALAUNCH_EV, p1, EMPTY);
}

void Probe_Cuda_Launch_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDALAUNCH_EV, EVT_END, EMPTY);
}

void Probe_Cuda_ConfigureCall_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDACONFIGCALL_EV, EVT_BEGIN, EMPTY);
}

void Probe_Cuda_ConfigureCall_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDACONFIGCALL_EV, EVT_END, EMPTY);
}

void Probe_Cuda_Memcpy_Entry (size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDAMEMCPY_EV, EVT_BEGIN, size);
}

void Probe_Cuda_Memcpy_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDAMEMCPY_EV, EVT_END, EMPTY); 
}

void Probe_Cuda_MemcpyAsync_Entry (size_t size)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDAMEMCPYASYNC_EV, EVT_BEGIN, size);
}

void Probe_Cuda_MemcpyAsync_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDAMEMCPYASYNC_EV, EVT_END, EMPTY); 
}

void Probe_Cuda_ThreadBarrier_Entry (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDATHREADBARRIER_EV, EVT_BEGIN, EMPTY);
}

void Probe_Cuda_ThreadBarrier_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDATHREADBARRIER_EV, EVT_END, EMPTY); 
}

void Probe_Cuda_StreamBarrier_Entry (unsigned threadid)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, CUDASTREAMBARRIER_EV, EVT_BEGIN, threadid);
}

void Probe_Cuda_StreamBarrier_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDASTREAMBARRIER_EV, EVT_END, EMPTY); 
}


void Probe_Cuda_DeviceReset_Enter (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDADEVICERESET_EV, EVT_BEGIN, EMPTY); 
}

void Probe_Cuda_DeviceReset_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDADEVICERESET_EV, EVT_END, EMPTY); 
}

void Probe_Cuda_ThreadExit_Enter (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDATHREADEXIT_EV, EVT_BEGIN, EMPTY); 
}

void Probe_Cuda_ThreadExit_Exit (void)
{
	DEBUG
	if (mpitrace_on && Extrae_get_trace_CUDA())
		TRACE_MISCEVENTANDCOUNTERS(TIME, CUDATHREADEXIT_EV, EVT_END, EMPTY); 
}
