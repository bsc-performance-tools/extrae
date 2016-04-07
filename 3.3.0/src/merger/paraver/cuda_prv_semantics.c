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

static int CUDA_Call (event_t* event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset )
{
	unsigned state;
	unsigned EvType, EvValue;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (event);
	EvValue = Get_EvValue (event);

	switch (EvType)
	{
		case CUDACONFIGCALL_EV:	
			state = STATE_OTHERS;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
		case CUDATHREADEXIT_EV:
		case CUDADEVICERESET_EV:
		case CUDALAUNCH_EV:
			state = STATE_OVHD;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
		case CUDASTREAMBARRIER_EV:
		case CUDATHREADBARRIER_EV:
			state = STATE_BARRIER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
		case CUDAMEMCPY_EV:
		case CUDAMEMCPYASYNC_EV:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state (cpu, ptask, task, thread, current_time);
	if (EvValue != EVT_END)
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDACALL_EV, EvType - CUDABASE_EV);
	else
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDACALL_EV, EVT_END);

	if (EvType == CUDAMEMCPY_EV || EvType == CUDAMEMCPYASYNC_EV)
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDAMEMCPY_SIZE_EV, Get_EvMiscParam(event));

	if (EvType == CUDALAUNCH_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDAFUNC_EV, EvValue);
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDAFUNC_LINE_EV, EvValue);
	}

	if (EvType == CUDASTREAMBARRIER_EV)
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDASTREAMBARRIER_THID_EV, 1+Get_EvMiscParam(event));

	return 0;
}

static int CUDA_GPU_Call (event_t *event, unsigned long long current_time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset)
{
	unsigned EvType, EvValue, state;
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (event);
	EvValue = Get_EvValue (event);

	switch (EvType)
	{
		case CUDAKERNEL_GPU_EV:
			state = STATE_RUNNING;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
		case CUDATHREADBARRIER_GPU_EV:
			state = STATE_BARRIER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
		case CUDAMEMCPYASYNC_GPU_EV:
		case CUDAMEMCPY_GPU_EV:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
		case CUDACONFIGKERNEL_GPU_EV:
			state = STATE_OTHERS;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state (cpu, ptask, task, thread, current_time);

	if (EvValue != EVT_END)
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDACALL_EV, EvType - CUDABASE_GPU_EV);
	else
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDACALL_EV, EVT_END);

	if (EvType == CUDAMEMCPY_GPU_EV || EvType == CUDAMEMCPYASYNC_EV)
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDAMEMCPY_SIZE_EV, Get_EvMiscParam(event));

	if (EvType == CUDAKERNEL_GPU_EV)
	{
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDAFUNC_EV, EvValue);
		trace_paraver_event (cpu, ptask, task, thread, current_time,
		  CUDAFUNC_LINE_EV, EvValue);
	}

	return 0;
}

SingleEv_Handler_t PRV_CUDA_Event_Handlers[] = {
	/* Host calls */
	{ CUDACONFIGCALL_EV, CUDA_Call },
	{ CUDALAUNCH_EV, CUDA_Call },
	{ CUDAMEMCPY_EV, CUDA_Call },
	{ CUDAMEMCPYASYNC_EV, CUDA_Call },
	{ CUDATHREADBARRIER_EV, CUDA_Call },
	{ CUDASTREAMBARRIER_EV, CUDA_Call },
	{ CUDADEVICERESET_EV, CUDA_Call },
	{ CUDATHREADEXIT_EV, CUDA_Call },
	/* Accelerator calls */
	{ CUDAKERNEL_GPU_EV, CUDA_GPU_Call },
	{ CUDACONFIGKERNEL_GPU_EV, CUDA_GPU_Call },
	{ CUDAMEMCPY_GPU_EV, CUDA_GPU_Call },
	{ CUDAMEMCPYASYNC_GPU_EV, CUDA_GPU_Call },
	{ CUDATHREADBARRIER_GPU_EV, CUDA_GPU_Call },
	{ NULL_EV, NULL }
};

