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
CUDA_Call(event_t *event, unsigned long long current_time, unsigned int cpu,
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
		case CUDASTREAMCREATE_VAL:
		case CUDASTREAMDESTROY_VAL:
		case CUDAEVENTRECORD_VAL:
			state = STATE_OTHERS;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDATHREADEXIT_VAL:
		case CUDADEVICERESET_VAL:
			state = STATE_OVHD;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDACONFIGCALL_VAL:
		case CUDALAUNCH_VAL:
			state = STATE_CONFACCEL;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDASTREAMBARRIER_VAL:
		case CUDATHREADBARRIER_VAL:
		case CUDAEVENTSYNCHRONIZE_VAL:
			state = STATE_BARRIER;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDAMEMCPY_VAL:
		case CUDAMEMCPYASYNC_VAL:
		case CUDAMEMSET_VAL:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDAMALLOC_VAL:
		case CUDAMALLOCPITCH_VAL:
		case CUDAFREE_VAL:
		case CUDAMALLOCARRAY_VAL:
		case CUDAFREEARRAY_VAL:
		case CUDAMALLOCHOST_VAL:
		case CUDAFREEHOST_VAL:
		case CUDAHOSTALLOC_VAL:
			state = STATE_ALLOCMEM;
			Switch_State(state, (EvMisc != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state(cpu, ptask, task, thread, current_time);
	trace_paraver_event(cpu, ptask, task, thread, current_time, CUDACALL_EV,
	    (EvMisc != EVT_END) ? EvValue : EVT_END);

	return 0;
}

static int
CUDA_Func_Event(event_t *event, unsigned long long current_time,
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
	trace_paraver_event(cpu, ptask, task, thread, current_time, EvType == CUDA_KERNEL_INST_EV ? CUDA_KERNEL_INST_LINE_EV : CUDA_KERNEL_EXEC_LINE_EV, EvValue);

	return 0;
}

static int
CUDA_Punctual_Event(event_t *event, unsigned long long current_time,
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
CUDA_GPU_Call (event_t *event, unsigned long long current_time,
    unsigned int cpu, unsigned int ptask, unsigned int task, unsigned int thread,
    FileSet_t *fset)
{
	unsigned EvMisc, state;
	UINT64 EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvValue = Get_EvValue (event);
	EvMisc  = Get_EvMiscParam(event);

	switch (EvValue)
	{
		case CUDAKERNEL_GPU_VAL:
			state = STATE_RUNNING;
			Switch_State (state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDASTREAMBARRIER_GPU_VAL:
		case CUDATHREADBARRIER_GPU_VAL:
			state = STATE_BARRIER;
			Switch_State (state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDAMEMCPYASYNC_GPU_VAL:
		case CUDAMEMCPY_GPU_VAL:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (EvMisc != EVT_END), ptask, task, thread);
			break;
		case CUDACONFIGKERNEL_GPU_VAL:
			state = STATE_CONFACCEL;
			Switch_State (state, (EvMisc != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state(cpu, ptask, task, thread, current_time);

	/*
	 * XXX
	 * Devices don't call cuda_launch. They actually run the kernel, thus we
	 * don't emit this event so the region is marked as Useful.
	 * XXX
	 */
	if (EvValue != CUDAKERNEL_GPU_VAL)
	{
		trace_paraver_event(cpu, ptask, task, thread, current_time, CUDACALL_EV,
		  (EvMisc != EVT_END) ? EvValue : EVT_END);
	}

	return 0;
}

SingleEv_Handler_t PRV_CUDA_Event_Handlers[] = {
	/* Host calls */
	{ CUDACALL_EV, CUDA_Call },
	{ CUDA_KERNEL_EXEC_EV, CUDA_Func_Event },
	{ CUDA_KERNEL_INST_EV, CUDA_Func_Event },
	{ CUDA_DYNAMIC_MEM_PTR_EV, CUDA_Punctual_Event },
	{ CUDA_DYNAMIC_MEM_SIZE_EV, CUDA_Punctual_Event },
	{ CUDASTREAMBARRIER_THID_EV, CUDA_Punctual_Event },
	{ CUDA_UNTRACKED_EV, CUDA_Punctual_Event },
	/* Accelerator calls */
	{ CUDACALLGPU_EV, CUDA_GPU_Call },
	{ NULL_EV, NULL }
};
