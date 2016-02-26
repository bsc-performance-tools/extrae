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
#include "communication_queues.h"
#include "trace_communication.h"
#include "addresses.h"
#include "options.h"
#include "labels.h"
#include "extrae_types.h"

#if USE_HARDWARE_COUNTERS
# include "HardwareCounters.h"
#endif

#include "addr2info.h" 

#if defined(PARALLEL_MERGE)
# include "parallel_merge_aux.h"
# include "timesync.h"
#endif

#include "events.h"
#include "paraver_state.h"

struct OpenCL_event_presency_label_st
{
	unsigned eventtype;
	unsigned present;
	char * description;
	int eventval;
}; 

#define MAX_OPENCL_TYPE_ENTRIES 52

static struct OpenCL_event_presency_label_st
 OpenCL_event_presency_label_host[MAX_OPENCL_TYPE_ENTRIES] = 
{
 { OPENCL_CLCREATEBUFFER_EV, FALSE, "clCreateBuffer", 1 },
 { OPENCL_CLCREATECOMMANDQUEUE_EV, FALSE, "clCreateCommandQueue", 2 },
 { OPENCL_CLCREATECONTEXT_EV, FALSE, "clCreateContext", 3 },
 { OPENCL_CLCREATECONTEXTFROMTYPE_EV, FALSE, "clCreateContextFromType", 4 },
 { OPENCL_CLCREATESUBBUFFER_EV, FALSE, "clCreateSubBuffer", 5 },
 { OPENCL_CLCREATEKERNEL_EV, FALSE, "clCreateKernel", 6 },
 { OPENCL_CLCREATEKERNELSINPROGRAM_EV, FALSE, "clCreateKernelsInProgram", 7 },
 { OPENCL_CLSETKERNELARG_EV, FALSE, "clSetKernelArg", 8 },
 { OPENCL_CLCREATEPROGRAMWITHSOURCE_EV, FALSE, "clCreateProgramWithSource", 9 },
 { OPENCL_CLCREATEPROGRAMWITHBINARY_EV, FALSE, "clCreateProgramWithBinary", 10 },
 { OPENCL_CLCREATEPROGRAMWITHBUILTINKERNELS_EV, FALSE, "clCreateProgramWithBuiltInKernels", 11 },
 { OPENCL_CLENQUEUEFILLBUFFER_EV, FALSE, "clEnqueueFillBuffer", 12 },
 { OPENCL_CLENQUEUECOPYBUFFER_EV, FALSE, "clEnqueueCopyBuffer", 13 },
 { OPENCL_CLENQUEUECOPYBUFFERRECT_EV, FALSE, "clEnqueueCopyBufferRect", 14 },
 { OPENCL_CLENQUEUENDRANGEKERNEL_EV, FALSE, "clEnqueueNDRangeKernel", 15 },
 { OPENCL_CLENQUEUETASK_EV, FALSE, "clEnqueueTask", 16 },
 { OPENCL_CLENQUEUENATIVEKERNEL_EV, FALSE, "clEnqueueNativeKernel", 17 },
 { OPENCL_CLENQUEUEREADBUFFER_EV, FALSE, "clEnqueueReadBuffer", 18 },
 { OPENCL_CLENQUEUEREADBUFFERRECT_EV, FALSE, "clEnqueueReadBufferRect", 19 },
 { OPENCL_CLENQUEUEWRITEBUFFER_EV, FALSE, "clEnqueueWriteBuffer", 20 },
 { OPENCL_CLENQUEUEWRITEBUFFERRECT_EV, FALSE, "clEnqueueWriteBufferRect", 21 },
 { OPENCL_CLBUILDPROGRAM_EV, FALSE, "clBuildProgram", 22 },
 { OPENCL_CLCOMPILEPROGRAM_EV, FALSE, "clCompileProgram", 23 },
 { OPENCL_CLLINKPROGRAM_EV, FALSE, "clLinkProgram", 24 },
 { OPENCL_CLFINISH_EV, FALSE, "clFinish", 25 },
 { OPENCL_CLFLUSH_EV, FALSE, "clFlush", 26 },
 { OPENCL_CLWAITFOREVENTS_EV, FALSE, "clWaitForEvents", 27 },
 { OPENCL_CLENQUEUEMARKERWITHWAITLIST_EV, FALSE, "clEnqueueMarkerWithWaitList", 28 },
 { OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV, FALSE, "clEnqueueBarrierWithWaitList", 29 },
 { OPENCL_CLENQUEUEMAPBUFFER_EV, FALSE, "clEnqueueMapBuffer", 30 },
 { OPENCL_CLENQUEUEUNMAPMEMOBJECT_EV, FALSE, "clEnqueueUnmapMemObject", 31 },
 { OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_EV, FALSE, "clEnqueueMigrateMemObjects", 32},
 { OPENCL_CLENQUEUEMARKER_EV, FALSE, "clEnqueueMarker", 33 },
 { OPENCL_CLENQUEUEBARRIER_EV, FALSE, "clEnqueueBarrier", 34 },
 { OPENCL_CLRETAINCOMMANDQUEUE_EV, FALSE, "clRetainCommandQueue", 35 },
 { OPENCL_CLRELEASECOMMANDQUEUE_EV, FALSE, "clReleaseCommandQueue", 36 },
 { OPENCL_CLRETAINCONTEXT_EV, FALSE, "clRetainContext", 37 },
 { OPENCL_CLRELEASECONTEXT_EV, FALSE, "clReleaseContext", 38 },
 { OPENCL_CLRETAINDEVICE_EV, FALSE, "clRetainDevice", 39 },
 { OPENCL_CLRELEASEDEVICE_EV, FALSE, "clReleaseDevice", 40 },
 { OPENCL_CLRETAINEVENT_EV, FALSE, "clRetainEvent", 41 },
 { OPENCL_CLRELEASEEVENT_EV, FALSE, "clReleaseEvent", 42 },
 { OPENCL_CLRETAINKERNEL_EV, FALSE, "clRetainKernel", 43 },
 { OPENCL_CLRELEASEKERNEL_EV, FALSE, "clReleaseKernel", 44 },
 { OPENCL_CLRETAINMEMOBJECT_EV, FALSE, "clRetainMemObject", 45 },
 { OPENCL_CLRELEASEMEMOBJECT_EV, FALSE, "clReleaseMemObject", 46 },
 { OPENCL_CLRETAINPROGRAM_EV, FALSE, "clRetainProgram", 47 },
 { OPENCL_CLRELEASEPROGRAM_EV, FALSE, "clReleaseProgram", 48 },
 { OPENCL_CLENQUEUEREADBUFFER_ASYNC_EV, FALSE, "clEnqueueReadBuffer (async)", 49 },
 { OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_EV, FALSE, "clEnqueueReadBufferRect (async)", 50 },
 { OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_EV, FALSE, "clEnqueueWriteBuffer (async)", 51 },
 { OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_EV, FALSE, "clEnqueueWriteBufferRect (async)", 52 }
};

static struct OpenCL_event_presency_label_st
 OpenCL_event_presency_label_acc[MAX_OPENCL_TYPE_ENTRIES] = 
{
 { 0, FALSE, "clCreateBuffer", 1 },
 { 0, FALSE, "clCreateCommandQueue", 2 },
 { 0, FALSE, "clCreateContext", 3 },
 { 0, FALSE, "clCreateContextFromType", 4 },
 { 0, FALSE, "clCreateSubBuffer", 5 },
 { 0, FALSE, "clCreateKernel", 6 },
 { 0, FALSE, "clCreateKernelsInProgram", 7 },
 { 0, FALSE, "clSetKernelArg", 8 },
 { 0, FALSE, "clCreateProgramWithSource", 9 },
 { 0, FALSE, "clCreateProgramWithBinary", 10 },
 { 0, FALSE, "clCreateProgramWithBuiltInKernels", 11 },
 { OPENCL_CLENQUEUEFILLBUFFER_ACC_EV, FALSE, "clEnqueueFillBuffer", 12 },
 { OPENCL_CLENQUEUECOPYBUFFER_ACC_EV, FALSE, "clEnqueueCopyBuffer", 13 },
 { OPENCL_CLENQUEUECOPYBUFFERRECT_ACC_EV, FALSE, "clEnqueueCopyBufferRect", 14 },
 { OPENCL_CLENQUEUENDRANGEKERNEL_ACC_EV, FALSE, "clEnqueueNDRangeKernel", 15 },
 { OPENCL_CLENQUEUETASK_ACC_EV, FALSE, "clEnqueueTask", 16 },
 { OPENCL_CLENQUEUENATIVEKERNEL_ACC_EV, FALSE, "clEnqueueNativeKernel", 17 },
 { OPENCL_CLENQUEUEREADBUFFER_ACC_EV, FALSE, "clEnqueueReadBuffer", 18 },
 { OPENCL_CLENQUEUEREADBUFFERRECT_ACC_EV, FALSE, "clEnqueueReadBufferRect", 19 },
 { OPENCL_CLENQUEUEWRITEBUFFER_ACC_EV, FALSE, "clEnqueueWriteBuffer", 20 },
 { OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV, FALSE, "clEnqueueWriteBufferRect", 21 },
 { 0, FALSE, "clBuildProgram", 22 },
 { 0, FALSE, "clCompileProgram", 23 },
 { 0, FALSE, "clLinkProgram", 24 },
 { 0, FALSE, "clFinish", 25 },
 { 0, FALSE, "clFlush", 26 },
 { 0, FALSE, "clWaitForEvents", 27 },
 { OPENCL_CLENQUEUEMARKERWITHWAITLIST_ACC_EV, FALSE, "clEnqueueMarkerWithWaitList", 28 },
 { OPENCL_CLENQUEUEBARRIERWITHWAITLIST_ACC_EV, FALSE, "clEnqueueBarrierWithWaitList", 29 },
 { OPENCL_CLENQUEUEMAPBUFFER_ACC_EV, FALSE, "clEnqueueMapBuffer", 30 },
 { OPENCL_CLENQUEUEUNMAPMEMOBJECT_ACC_EV, FALSE, "clEnqueueUnmapMemObject", 31 },
 { OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_ACC_EV, FALSE, "clEnqueueMigrateMemObjects", 32},
 { OPENCL_CLENQUEUEMARKER_ACC_EV, FALSE, "clEnqueueMarker", 33 },
 { OPENCL_CLENQUEUEBARRIER_ACC_EV, FALSE, "clEnqueueBarrier", 34 },
 { 0, FALSE, "clRetainCommandQueue", 35 },
 { 0, FALSE, "clReleaseCommandQueue", 36 },
 { 0, FALSE, "clRetainContext", 37 },
 { 0, FALSE, "clReleaseContext", 38 },
 { 0, FALSE, "clRetainDevice", 39 },
 { 0, FALSE, "clReleaseDevice", 40 },
 { 0, FALSE, "clRetainEvent", 41 },
 { 0, FALSE, "clReleaseEvent", 42 },
 { 0, FALSE, "clRetainKernel", 43 },
 { 0, FALSE, "clReleaseKernel", 44 },
 { 0, FALSE, "clRetainMemObject", 45 },
 { 0, FALSE, "clReleaseMemObject", 46 },
 { 0, FALSE, "clRetainProgram", 47 },
 { 0, FALSE, "clReleaseProgram", 48 },
 { OPENCL_CLENQUEUEREADBUFFER_ASYNC_ACC_EV, FALSE, "clEnqueueReadBuffer (async)", 49 },
 { OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_ACC_EV, FALSE, "clEnqueueReadBufferRect (async)", 50 },
 { OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_ACC_EV, FALSE, "clEnqueueWriteBuffer (async)", 51 },
 { OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_ACC_EV, FALSE, "clEnqueueWriteBufferRect (async)", 52 }
};


#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_OpenCL_Operations (void)
{
	int res;
	int i, tmp_in[MAX_OPENCL_TYPE_ENTRIES], tmp_out[MAX_OPENCL_TYPE_ENTRIES];

	/* Share host-side calls */
	for (i = 0; i < MAX_OPENCL_TYPE_ENTRIES; i++)
		tmp_in[i] = OpenCL_event_presency_label_host[i].present;

	res = MPI_Reduce (tmp_in, tmp_out, MAX_OPENCL_TYPE_ENTRIES, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing OpenCL enabled operations");

	for (i = 0; i < MAX_OPENCL_TYPE_ENTRIES; i++)
		OpenCL_event_presency_label_host[i].present = tmp_out[i];

	/* Share accelerator-side calls */
	for (i = 0; i < MAX_OPENCL_TYPE_ENTRIES; i++)
		tmp_in[i] = OpenCL_event_presency_label_acc[i].present;

	res = MPI_Reduce (tmp_in, tmp_out, MAX_OPENCL_TYPE_ENTRIES, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing OpenCL enabled operations");

	for (i = 0; i < MAX_OPENCL_TYPE_ENTRIES; i++)
		OpenCL_event_presency_label_acc[i].present = tmp_out[i];
}
#endif /* PARALLEL_MERGE */

void Enable_OpenCL_Operation (unsigned evttype)
{
	unsigned u;
	struct OpenCL_event_presency_label_st *table;

	if (evttype >= OPENCL_BASE_TYPE_EV && evttype < OPENCL_BASE_TYPE_ACC_EV)
		table = OpenCL_event_presency_label_host;
	else
		table = OpenCL_event_presency_label_acc;

	for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
		if (table[u].eventtype == evttype)
		{
			table[u].present = TRUE;
			break;
		}
}

int Translate_OpenCL_Operation (unsigned in_evttype, 
	unsigned long long in_evtvalue, unsigned *out_evttype,
	unsigned long long *out_evtvalue)
{
	unsigned u;
	struct OpenCL_event_presency_label_st *table;
	unsigned out_type;

	if (in_evttype >= OPENCL_BASE_TYPE_EV && in_evttype < OPENCL_BASE_TYPE_ACC_EV)
	{
		table = OpenCL_event_presency_label_host;
		out_type = OPENCL_BASE_TYPE_EV;
	}
	else
	{
		table = OpenCL_event_presency_label_acc;
		out_type = OPENCL_BASE_TYPE_ACC_EV;
	}
	
	for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
		if (table[u].eventtype == in_evttype)
		{
			*out_evttype = out_type;
			if (in_evtvalue != 0)
				*out_evtvalue = table[u].eventval;
			else
				*out_evtvalue = 0;
			return TRUE;
		}

	return FALSE;
}

void WriteEnabled_OpenCL_Operations (FILE * fd)
{
	unsigned u;
	int anypresent = FALSE;
	int memtransfersizepresent = FALSE;
	int clfinishpresent = FALSE;

	for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
	{
		anypresent = OpenCL_event_presency_label_host[u].present || anypresent;

		if (OpenCL_event_presency_label_host[u].present && (
		      OpenCL_event_presency_label_host[u].eventtype == OPENCL_CLENQUEUEREADBUFFER_EV ||
		      OpenCL_event_presency_label_host[u].eventtype == OPENCL_CLENQUEUEREADBUFFERRECT_EV ||
		      OpenCL_event_presency_label_host[u].eventtype == OPENCL_CLENQUEUEWRITEBUFFER_EV ||
		      OpenCL_event_presency_label_host[u].eventtype == OPENCL_CLENQUEUEWRITEBUFFERRECT_EV )
		   )
			memtransfersizepresent = TRUE;

		if (OpenCL_event_presency_label_host[u].present && (
		     OpenCL_event_presency_label_host[u].eventtype == OPENCL_CLFINISH_EV))
			clfinishpresent = TRUE;
	}

	if (anypresent)
	{

		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d    %d    %s\n", 0, OPENCL_BASE_TYPE_EV, "Host OpenCL call");
		fprintf (fd, "VALUES\n");
		fprintf (fd, "0 Outside OpenCL\n");
		for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
			if (OpenCL_event_presency_label_host[u].present)
				fprintf (fd, "%d %s\n", 
					OpenCL_event_presency_label_host[u].eventval,
					OpenCL_event_presency_label_host[u].description);
		LET_SPACES(fd);

		if (memtransfersizepresent)
		{
			fprintf (fd, "EVENT_TYPE\n"
		              "%d   %d    OpenCL transfer size\n"
		              "\n",
		              0, OPENCL_CLMEMOP_SIZE_EV);
		}
	}

	anypresent = FALSE;
	for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
		anypresent = OpenCL_event_presency_label_acc[u].present || anypresent;

	if (anypresent)
	{
		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d    %d    %s\n", 0, OPENCL_BASE_TYPE_ACC_EV, "Accelerator OpenCL call");
		fprintf (fd, "VALUES\n");
		fprintf (fd, "0 Outside OpenCL\n");
		for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
			if (OpenCL_event_presency_label_acc[u].present &&
			    OpenCL_event_presency_label_acc[u].eventtype != 0)
				fprintf (fd, "%d %s\n", 
					OpenCL_event_presency_label_acc[u].eventval,
					OpenCL_event_presency_label_acc[u].description);
		LET_SPACES(fd);
	}

	if (clfinishpresent)
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d    %d    Synchronized command queue (on thread)\n"
                     "\n",
                     0, OPENCL_CLFINISH_THID_EV);
	}
}

