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
#include "misc_prv_events.h"
#include "semantics.h"
#include "paraver_generator.h"
#include "options.h"
#include "extrae_types.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#include "events.h"
#include "paraver_state.h"

#include "opencl_prv_events.h"

static int OpenCL_Acc_Call (event_t * event, unsigned long long time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset )
{
	unsigned state;
	unsigned EvType, nEvType;
	unsigned long long EvValue, nEvValue;
	
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (event);
	EvValue = Get_EvValue (event);

	switch (EvType)
	{
		case OPENCL_CLENQUEUEFILLBUFFER_ACC_EV:
		case OPENCL_CLENQUEUECOPYBUFFER_ACC_EV:
		case OPENCL_CLENQUEUECOPYBUFFERRECT_ACC_EV:
		case OPENCL_CLENQUEUEREADBUFFER_ACC_EV:
		case OPENCL_CLENQUEUEWRITEBUFFER_ACC_EV:
		case OPENCL_CLENQUEUEREADBUFFERRECT_ACC_EV:
		case OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV:
		case OPENCL_CLENQUEUEMAPBUFFER_ACC_EV:
		case OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_ACC_EV:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;

		case OPENCL_CLENQUEUEBARRIERWITHWAITLIST_ACC_EV:
		case OPENCL_CLENQUEUEBARRIER_ACC_EV:
			state = STATE_BARRIER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;

		case OPENCL_CLENQUEUENDRANGEKERNEL_ACC_EV:
		case OPENCL_CLENQUEUETASK_ACC_EV:
		case OPENCL_CLENQUEUENATIVEKERNEL_ACC_EV:
			state = STATE_RUNNING;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;

		default:
			state = STATE_OVHD;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state (cpu, ptask, task, thread, time);

	Translate_OpenCL_Operation (EvType, EvValue, &nEvType, &nEvValue);
	trace_paraver_event (cpu, ptask, task, thread, time, nEvType, nEvValue);

	if (EvType == OPENCL_CLENQUEUENDRANGEKERNEL_ACC_EV ||
	  EvType == OPENCL_CLENQUEUETASK_ACC_EV)
	{
		unsigned long long EvParam;
		EvParam = Get_EvParam (event);
		trace_paraver_event (cpu, ptask, task, thread, time,
		  OPENCL_KERNEL_NAME_EV, EvParam);
	}

	if (EvType == OPENCL_CLENQUEUEREADBUFFER_ACC_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFER_ACC_EV ||
	    EvType == OPENCL_CLENQUEUEREADBUFFERRECT_ACC_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV ||
	    EvType == OPENCL_CLENQUEUEREADBUFFER_ASYNC_ACC_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_ACC_EV ||
	    EvType == OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_ACC_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_ACC_EV)
	{
		unsigned long long EvParam;
		EvParam = Get_EvParam (event);
		trace_paraver_event (cpu, ptask, task, thread, time,
		  OPENCL_CLMEMOP_SIZE_EV, EvParam);
	}

	return 0;
}


static int OpenCL_Host_Call (event_t * event, unsigned long long time,
	unsigned int cpu, unsigned int ptask, unsigned int task,
	unsigned int thread, FileSet_t *fset )
{
	unsigned state;
	unsigned EvType, nEvType;
	unsigned long long EvValue, nEvValue;
	
	UNREFERENCED_PARAMETER(fset);

	EvType  = Get_EvEvent (event);
	EvValue = Get_EvValue (event);

	switch (EvType)
	{
		case OPENCL_CLENQUEUEFILLBUFFER_EV:
		case OPENCL_CLENQUEUECOPYBUFFER_EV:
		case OPENCL_CLENQUEUECOPYBUFFERRECT_EV:
		case OPENCL_CLENQUEUEREADBUFFER_EV:
		case OPENCL_CLENQUEUEWRITEBUFFER_EV:
		case OPENCL_CLENQUEUEREADBUFFERRECT_EV:
		case OPENCL_CLENQUEUEWRITEBUFFERRECT_EV:
		case OPENCL_CLENQUEUEMAPBUFFER_EV:
		case OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_EV:
			state = STATE_MEMORY_XFER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;

		case OPENCL_CLWAITFOREVENTS_EV:
		case OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV:
		case OPENCL_CLENQUEUEBARRIER_EV:
		case OPENCL_CLFINISH_EV:
			state = STATE_BARRIER;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;

		default:
			state = STATE_OVHD;
			Switch_State (state, (EvValue != EVT_END), ptask, task, thread);
			break;
	}

	trace_paraver_state (cpu, ptask, task, thread, time);

	Translate_OpenCL_Operation (EvType, EvValue, &nEvType, &nEvValue);
	trace_paraver_event (cpu, ptask, task, thread, time, nEvType, nEvValue);

	if (EvType == OPENCL_CLENQUEUENDRANGEKERNEL_EV ||
	  EvType == OPENCL_CLENQUEUETASK_EV)
	{
		unsigned long long EvParam;
		EvParam = Get_EvParam (event);
		trace_paraver_event (cpu, ptask, task, thread, time,
		  OPENCL_KERNEL_NAME_EV, EvParam);
	}

	if (EvType == OPENCL_CLENQUEUEREADBUFFER_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFER_EV ||
	    EvType == OPENCL_CLENQUEUEREADBUFFERRECT_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFERRECT_EV ||
	    EvType == OPENCL_CLENQUEUEREADBUFFER_ASYNC_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_EV ||
	    EvType == OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_EV ||
	    EvType == OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_EV)
	{
		unsigned long long EvParam;
		EvParam = Get_EvParam (event);
		trace_paraver_event (cpu, ptask, task, thread, time,
		  OPENCL_CLMEMOP_SIZE_EV, EvParam);
	}

	if (EvType == OPENCL_CLFINISH_EV && EvValue != EVT_END)
		trace_paraver_event (cpu, ptask, task, thread, time,
		  OPENCL_CLFINISH_THID_EV, 1+Get_EvMiscParam(event));

	return 0;
}

RangeEv_Handler_t PRV_OpenCL_Event_Handlers[] = {
	/* Host side */
	{ OPENCL_CLCREATEBUFFER_EV, OPENCL_CLRELEASEPROGRAM_EV, OpenCL_Host_Call },
	{ OPENCL_CLENQUEUEREADBUFFER_ASYNC_EV, OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_EV, OpenCL_Host_Call },

	/* Accelerator-side */
	{ OPENCL_CLENQUEUEFILLBUFFER_ACC_EV, OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV, OpenCL_Acc_Call },
	{ OPENCL_CLENQUEUEMARKERWITHWAITLIST_ACC_EV, OPENCL_CLENQUEUEBARRIER_ACC_EV, OpenCL_Acc_Call },
	{ OPENCL_CLENQUEUEREADBUFFER_ASYNC_ACC_EV, OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_ACC_EV, OpenCL_Acc_Call },

	{ NULL_EV, NULL_EV, NULL }
};

