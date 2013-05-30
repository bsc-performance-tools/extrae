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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.3/src/merger/paraver/mpi_prv_semantics.c $
 | @last_commit: $Date: 2013-05-23 18:04:22 +0200 (dj, 23 mai 2013) $
 | @version:     $Revision: 1761 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: mpi_prv_semantics.c 1761 2013-05-23 16:04:22Z harald $";

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

#define MAX_OPENCL_TYPE_ENTRIES 29

static struct OpenCL_event_presency_label_st OpenCL_event_presency_label[MAX_OPENCL_TYPE_ENTRIES] = 
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
 { OPENCL_CLENQUEUEBARRIERWITHWAITLIST_EV, FALSE, "clEnqueueBarrierWithWaitList", 29 }
};

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_OpenCL_Operations (void)
{
	int res;
	int i, tmp_in[MAX_OPENCL_TYPE_ENTRIES], tmp_out[MAX_OPENCL_TYPE_ENTRIES];

	for (i = 0; i < MAX_OPENCL_TYPE_ENTRIES; i++)
		tmp_in[i] = OpenCL_event_presency_label[i].present;

	res = MPI_Reduce (tmp_in, tmp_out, MAX_OPENCL_TYPE_ENTRIES, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing OpenCL enabled operations");

	for (i = 0; i < MAX_OPENCL_TYPE_ENTRIES; i++)
		OpenCL_event_presency_label[i].present = tmp_out[i];
}
#endif /* PARALLEL_MERGE */

void Enable_OpenCL_Operation (unsigned evttype)
{
	unsigned u;

	for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
		if (OpenCL_event_presency_label[u].eventtype == evttype)
		{
			OpenCL_event_presency_label[u].present = TRUE;
			break;
		}
}

int Translate_OpenCL_Operation (unsigned in_evttype, 
	unsigned long long in_evtvalue, unsigned *out_evttype,
	unsigned long long *out_evtvalue)
{
	unsigned u;
	
	for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
		if (OpenCL_event_presency_label[u].eventtype == in_evttype)
		{
			*out_evttype = OPENCL_BASE_TYPE_EV;
			if (in_evtvalue != 0)
				*out_evtvalue = OpenCL_event_presency_label[u].eventval;
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

	for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES && !anypresent; u++)
		anypresent = OpenCL_event_presency_label[u].present;

	if (anypresent)
	{

		fprintf (fd, "EVENT_TYPE\n");
		fprintf (fd, "%d    %d    %s\n", 0, OPENCL_BASE_TYPE_EV, "OpenCL call");
		fprintf (fd, "VALUES\n");
		fprintf (fd, "0 Outside OpenCL\n");
		for (u = 0; u < MAX_OPENCL_TYPE_ENTRIES; u++)
			if (OpenCL_event_presency_label[u].present)
				fprintf (fd, "%d %s\n", 
					OpenCL_event_presency_label[u].eventval,
					OpenCL_event_presency_label[u].description);
		LET_SPACES(fd);
	}
}

