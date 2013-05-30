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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.3/src/tracer/probes/CUDA/cuda_probe.c $
 | @last_commit: $Date: 2011-10-17 16:29:40 +0200 (dl, 17 oct 2011) $
 | @version:     $Revision: 785 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: cuda_probe.c 785 2011-10-17 14:29:40Z harald $";

#include "threadid.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "opencl_probe.h"

void Extrae_Probe_clCreateBuffer_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateBuffer_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateCommandQueue_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATECOMMANDQUEUE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateCommandQueue_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATECOMMANDQUEUE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateContext_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATECONTEXT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateContext_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATECONTEXT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateContextFromType_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATECONTEXTFROMTYPE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateContextFromType_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATECONTEXTFROMTYPE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateSubBuffer_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATESUBBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateSubBuffer_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATESUBBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateKernel_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEKERNEL_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateKernel_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateKernelsInProgram_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEKERNELSINPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateKernelsInProgram_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEKERNELSINPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clSetKernelArg_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLSETKERNELARG_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clSetKernelArg_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLSETKERNELARG_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateProgramWithSource_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEPROGRAMWITHSOURCE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateProgramWithSource_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEPROGRAMWITHSOURCE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateProgramWithBinary_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEPROGRAMWITHBINARY_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateProgramWithBinary_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEPROGRAMWITHBINARY_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateProgramWithBuiltInKernels_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEPROGRAMWITHBUILTINKERNELS_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateProgramWithBuiltInKernels_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEPROGRAMWITHBUILTINKERNELS_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueFillBuffer_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEFILLBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueFillBuffer_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEFILLBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueCopyBuffer_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUECOPYBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueCopyBuffer_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUECOPYBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueCopyBufferRect_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUECOPYBUFFERRECT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueCopyBufferRect_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUECOPYBUFFERRECT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueNDRangeKernel_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUENDRANGEKERNEL_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueNDRangeKernel_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUENDRANGEKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueTask_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUETASK_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueTask_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUETASK_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueNativeKernel_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUENATIVEKERNEL_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueNativeKernel_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUENATIVEKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueReadBuffer_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEREADBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueReadBuffer_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEREADBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueReadBufferRect_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEREADBUFFERRECT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueReadBufferRect_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEREADBUFFERRECT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueWriteBuffer_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEWRITEBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueWriteBuffer_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEWRITEBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueWriteBufferRect_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEWRITEBUFFERRECT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueWriteBufferRect_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEWRITEBUFFERRECT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clBuildProgram_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLBUILDPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clBuildProgram_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLBUILDPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCompileProgram_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCOMPILEPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCompileProgram_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCOMPILEPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clLinkProgram_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLLINKPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clLinkProgram_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLLINKPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clFinish_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLFINISH_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clFinish_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLFINISH_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clFlush_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLFLUSH_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clFlush_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLFLUSH_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clWaitForEvents_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLWAITFOREVENTS_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clWaitForEvents_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLWAITFOREVENTS_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueMarkerWithWaitList_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEMARKERWITHWAITLIST_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueMarkerWithWaitList_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEMARKERWITHWAITLIST_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueBarrierWithWaitList_Enter (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueBarrierWithWaitList_Exit (void)
{
	if (EXTRAE_ON() && EXTRAE_TRACING_OPENCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

