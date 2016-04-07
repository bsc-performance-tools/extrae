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
#include "opencl_probe.h"

static int trace_opencl = TRUE;

void Extrae_set_trace_OpenCL (int b)
{ trace_opencl = b; }

int Extrae_get_trace_OpenCL (void)
{ return trace_opencl; }

#if 0
# define DEBUG fprintf (stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__);
#else
# define DEBUG
#endif

void Extrae_Probe_clCreateBuffer_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateBuffer_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateCommandQueue_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATECOMMANDQUEUE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateCommandQueue_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATECOMMANDQUEUE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateContext_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATECONTEXT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateContext_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATECONTEXT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateContextFromType_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATECONTEXTFROMTYPE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateContextFromType_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATECONTEXTFROMTYPE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateSubBuffer_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATESUBBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateSubBuffer_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATESUBBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateKernel_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEKERNEL_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateKernel_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateKernelsInProgram_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEKERNELSINPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateKernelsInProgram_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEKERNELSINPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clSetKernelArg_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLSETKERNELARG_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clSetKernelArg_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLSETKERNELARG_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateProgramWithSource_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEPROGRAMWITHSOURCE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateProgramWithSource_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEPROGRAMWITHSOURCE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateProgramWithBinary_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEPROGRAMWITHBINARY_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateProgramWithBinary_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEPROGRAMWITHBINARY_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCreateProgramWithBuiltInKernels_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCREATEPROGRAMWITHBUILTINKERNELS_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCreateProgramWithBuiltInKernels_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCREATEPROGRAMWITHBUILTINKERNELS_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueFillBuffer_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEFILLBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueFillBuffer_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEFILLBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueCopyBuffer_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUECOPYBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueCopyBuffer_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUECOPYBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueCopyBufferRect_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUECOPYBUFFERRECT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueCopyBufferRect_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUECOPYBUFFERRECT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueNDRangeKernel_Enter (unsigned long long KID)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUENDRANGEKERNEL_EV, EVT_BEGIN, KID);
	}
}

void Extrae_Probe_clEnqueueNDRangeKernel_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUENDRANGEKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueTask_Enter (unsigned long long KID)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUETASK_EV, EVT_BEGIN, KID);
	}
}

void Extrae_Probe_clEnqueueTask_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUETASK_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueNativeKernel_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUENATIVEKERNEL_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueNativeKernel_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUENATIVEKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueReadBuffer_Enter (int sync, size_t size)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME,
		  sync?OPENCL_CLENQUEUEREADBUFFER_EV:OPENCL_CLENQUEUEREADBUFFER_ASYNC_EV,
		  EVT_BEGIN, size);
	}
}

void Extrae_Probe_clEnqueueReadBuffer_Exit (int sync)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME,
		  sync?OPENCL_CLENQUEUEREADBUFFER_EV:OPENCL_CLENQUEUEREADBUFFER_ASYNC_EV,
		  EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueReadBufferRect_Enter (int sync)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME,
		  sync?OPENCL_CLENQUEUEREADBUFFERRECT_EV:OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_EV,
		  EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueReadBufferRect_Exit (int sync)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME,
		  sync?OPENCL_CLENQUEUEREADBUFFERRECT_EV:OPENCL_CLENQUEUEREADBUFFERRECT_EV,
		  EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueWriteBuffer_Enter (int sync, size_t size)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME,
		  sync?OPENCL_CLENQUEUEWRITEBUFFER_EV:OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_EV,
		  EVT_BEGIN, size);
	}
}

void Extrae_Probe_clEnqueueWriteBuffer_Exit (int sync)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME,
		  sync?OPENCL_CLENQUEUEWRITEBUFFER_EV:OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_EV,
		  EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueWriteBufferRect_Enter (int sync)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME,
		  sync?OPENCL_CLENQUEUEWRITEBUFFERRECT_EV:OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_EV,
		  EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueWriteBufferRect_Exit (int sync)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME,
		  sync?OPENCL_CLENQUEUEWRITEBUFFERRECT_EV:OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_EV,
		  EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clBuildProgram_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLBUILDPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clBuildProgram_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLBUILDPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clCompileProgram_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLCOMPILEPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clCompileProgram_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLCOMPILEPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clLinkProgram_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLLINKPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clLinkProgram_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLLINKPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clFinish_Enter (unsigned threadid)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLFINISH_EV,
		  EVT_BEGIN, threadid);
	}
}

void Extrae_Probe_clFinish_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLFINISH_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clFlush_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLFLUSH_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clFlush_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLFLUSH_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clWaitForEvents_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLWAITFOREVENTS_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clWaitForEvents_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLWAITFOREVENTS_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueMarkerWithWaitList_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEMARKERWITHWAITLIST_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueMarkerWithWaitList_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEMARKERWITHWAITLIST_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueBarrierWithWaitList_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueBarrierWithWaitList_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueMarker_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEMARKER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueMarker_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEMARKER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}


void Extrae_Probe_clEnqueueBarrier_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEBARRIER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueBarrier_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEBARRIER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clEnqueueUnmapMemObject_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEUNMAPMEMOBJECT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueUnmapMemObject_Exit(void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEUNMAPMEMOBJECT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clEnqueueMapBuffer_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEMAPBUFFER_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueMapBuffer_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEMAPBUFFER_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clEnqueueMigrateMemObjects_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clEnqueueMigrateMemObjects_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clRetainCommandQueue_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRETAINCOMMANDQUEUE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clRetainCommandQueue_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRETAINCOMMANDQUEUE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clReleaseCommandQueue_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRELEASECOMMANDQUEUE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clReleaseCommandQueue_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRELEASECOMMANDQUEUE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clRetainContext_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRETAINCONTEXT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clRetainContext_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRETAINCONTEXT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clReleaseContext_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRELEASECONTEXT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clReleaseContext_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRELEASECONTEXT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clRetainDevice_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRETAINDEVICE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clRetainDevice_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRETAINDEVICE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clReleaseDevice_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRELEASEDEVICE_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clReleaseDevice_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRELEASEDEVICE_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clRetainEvent_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRETAINEVENT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clRetainEvent_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRETAINEVENT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clReleaseEvent_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRELEASEEVENT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clReleaseEvent_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRELEASEEVENT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clRetainKernel_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRETAINKERNEL_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clRetainKernel_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRETAINKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clReleaseKernel_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRELEASEKERNEL_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clReleaseKernel_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRELEASEKERNEL_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clRetainMemObject_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRETAINMEMOBJECT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clRetainMemObject_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRETAINMEMOBJECT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clReleaseMemObject_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRELEASEMEMOBJECT_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clReleaseMemObject_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRELEASEMEMOBJECT_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clRetainProgram_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRETAINPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clRetainProgram_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRETAINPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}

void Extrae_Probe_clReleaseProgram_Enter (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		Backend_Enter_Instrumentation (2);
		TRACE_MISCEVENTANDCOUNTERS (LAST_READ_TIME, OPENCL_CLRELEASEPROGRAM_EV, EVT_BEGIN, EMPTY);
	}
}

void Extrae_Probe_clReleaseProgram_Exit (void)
{
	DEBUG
	if (EXTRAE_ON() && Extrae_get_trace_OpenCL())
	{
		TRACE_MISCEVENTANDCOUNTERS (TIME, OPENCL_CLRELEASEPROGRAM_EV, EVT_END, EMPTY);
		Backend_Leave_Instrumentation ();
	}
}
