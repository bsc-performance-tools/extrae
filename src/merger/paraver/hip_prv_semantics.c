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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif

#include "file_set.h"
#include "object_tree.h"
#include "trace_to_prv.h"
#include "semantics.h"
#include "paraver_state.h"
#include "paraver_generator.h"
#include "addresses.h"
#include "options.h"

#include "record.h"
#include "events.h"

static int
HIP_Call(event_t *event, unsigned long long current_time, unsigned int cpu,
    unsigned int ptask, unsigned int task, unsigned int thread, FileSet_t *fset)
{
	unsigned int state;
	unsigned int EvMisc;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvValue = Get_EvValue(event);
	EvMisc  = Get_EvMiscParam(event);

	switch (EvValue)
	{
		case HIPSTREAMCREATE_VAL:
		case HIPSTREAMDESTROY_VAL:
		case HIPEVENTRECORD_VAL:
			state = STATE_OTHERS;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case HIPTHREADEXIT_VAL:
		case HIPDEVICERESET_VAL:
			state = STATE_OVHD;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case HIPCONFIGCALL_VAL:
		case HIPLAUNCH_VAL:
			state = STATE_CONFACCEL;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case HIPSTREAMBARRIER_VAL:
		case HIPTHREADBARRIER_VAL:
		case HIPEVENTSYNCHRONIZE_VAL:
			state = STATE_BARRIER;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case HIPMEMCPY_VAL:
		case HIPMEMCPYASYNC_VAL:
		case HIPMEMSET_VAL:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case HIPMALLOC_VAL:
		case HIPMALLOCPITCH_VAL:
		case HIPFREE_VAL:
		case HIPMALLOCARRAY_VAL:
		case HIPFREEARRAY_VAL:
		case HIPMALLOCHOST_VAL:
		case HIPFREEHOST_VAL:
		case HIPHOSTALLOC_VAL:
			state = STATE_ALLOCMEM;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state(cpu, ptask, task, thread, current_time);
	trace_paraver_event(cpu, ptask, task, thread, current_time, HIPCALL_EV,
	    (EvMisc != EVT_END) ? EvValue : EVT_END);

	return 0;
}

static int
HIP_Kernel_Inst_Event(event_t *event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	unsigned int state;
	unsigned int EvType;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(state);
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent(event);
	EvValue = Get_EvValue(event);


	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, EvValue);
	trace_paraver_event(cpu, ptask, task, thread, current_time, HIP_KERNEL_INST_LINE_EV, EvValue);

	return 0;
}

static int
HIP_Kernel_Exec_Event(event_t *event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	unsigned int state;
	unsigned int EvType;
	unsigned int blocksPerGrid, threadsPerBlock;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(state);
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent(event);
	EvValue = Get_EvValue(event);
	blocksPerGrid = Get_GPUEvGridSize(event);
	threadsPerBlock = Get_GPUEvBlockSize(event);

	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, EvValue);
	trace_paraver_event(cpu, ptask, task, thread, current_time, HIP_KERNEL_EXEC_LINE_EV, EvValue);

	if(EvValue != EVT_END)
	{
		trace_paraver_event(cpu, ptask, task, thread, current_time, HIP_KERNEL_BLOCKS_PER_GRID, blocksPerGrid);
		trace_paraver_event(cpu, ptask, task, thread, current_time, HIP_KERNEL_THREADS_PER_BLOCK, threadsPerBlock);
	}

	return 0;
}

static int
HIP_Punctual_Event(event_t *event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	unsigned int state;
	unsigned int EvType;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(state);
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent(event);
	EvValue = Get_EvValue(event);

	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType, EvValue);

	return 0;
}

static int
HIP_GPU_Call (event_t *event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	unsigned state, beginEV;
	size_t size;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvValue = Get_EvValue (event);
	
	size = Get_GPUEvMemSize(event);
	beginEV = Get_GPUEvBegin(event);

	switch (EvValue)
	{
		case HIPKERNEL_GPU_VAL:
			state = STATE_RUNNING;
			Switch_State (state, (beginEV != EVT_END), ptask, task, thread);
			break;
		case HIPMEMCPYASYNC_GPU_VAL:
		case HIPMEMCPY_GPU_VAL:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (beginEV != EVT_END), ptask, task, thread);
			break;
		case HIPCONFIGKERNEL_GPU_VAL:
			state = STATE_CONFACCEL;
			Switch_State (state, (beginEV != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state(cpu, ptask, task, thread, current_time);

	/*
	 * XXX
	 * Devices don't call hip_launch. They actually run the kernel, thus we
	 * don't emit this event so the region is marked as Useful.
	 * XXX
	 */
	if (EvValue == HIPMEMCPY_GPU_VAL || EvValue == HIPMEMCPYASYNC_GPU_VAL)
	{
		trace_paraver_event(cpu, ptask, task, thread, current_time, HIP_MEMORY_TRANSFER, (beginEV != EVT_END) ? EvValue : EVT_END);
		if(beginEV != EVT_END)
		{
			trace_paraver_event(cpu, ptask, task, thread, current_time, HIP_DYNAMIC_MEM_SIZE_EV, size);
		}
	}

	return 0;
}

SingleEv_Handler_t PRV_HIP_Event_Handlers[] = {
	/* Host calls */
	{ HIPCALL_EV, HIP_Call },
	{ HIP_KERNEL_EXEC_EV, HIP_Kernel_Exec_Event },
	{ HIP_KERNEL_INST_EV, HIP_Kernel_Inst_Event },
	{ HIP_DYNAMIC_MEM_PTR_EV, HIP_Punctual_Event },
	{ HIP_DYNAMIC_MEM_SIZE_EV, HIP_Punctual_Event },
	{ HIPSTREAMBARRIER_THID_EV, HIP_Punctual_Event },
	{ HIP_UNTRACKED_EV, HIP_Punctual_Event },
	/* Accelerator calls */
	{ HIPCALLGPU_EV, HIP_GPU_Call },
	{ NULL_EV, NULL }
};
