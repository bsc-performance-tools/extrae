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

#include "events.h"
#include "mpi2out.h"
#include "options.h"

enum {
	HIPLAUNCH_INDEX,
	HIPCONFIGCALL_INDEX,
	HIPMEMCPY_INDEX,
	HIPTHREADBARRIER_INDEX,
	HIPSTREAMBARRIER_INDEX,
	HIPMEMCPYASYNC_INDEX,
	HIPTHREADEXIT_INDEX,
	HIPDEVICERESET_INDEX,
	HIPSTREAMCREATE_INDEX,
	HIPSTREAMDESTROY_INDEX,
	HIPMALLOC_INDEX,
	HIPHOSTALLOC_INDEX,
	HIPMEMSET_INDEX,
	HIPEVENTRECORD_INDEX,
	HIPEVENTSYNCHRONIZE_INDEX,
	HIPUNTRACKED_INDEX,
	MAX_HIP_INDEX
};


static int inuse[MAX_HIP_INDEX] = { FALSE };

void Enable_HIP_Operation (INT32 type, UINT64 value)
{
	if (type == HIP_UNTRACKED_EV){
			inuse[HIPUNTRACKED_INDEX] = TRUE;
		}
	else if(type == HIPCALL_EV || type == HIPCALLGPU_EV){
		if (value == HIPLAUNCH_VAL || value == HIPKERNEL_GPU_VAL)
			inuse[HIPLAUNCH_INDEX] = TRUE;
		else if (value == HIPMEMCPY_VAL || value == HIPMEMCPY_GPU_VAL)
			inuse[HIPMEMCPY_INDEX] = TRUE;
		else if (value == HIPSTREAMBARRIER_VAL)
			inuse[HIPSTREAMBARRIER_INDEX] = TRUE;
		else if (value == HIPTHREADBARRIER_VAL)
			inuse[HIPTHREADBARRIER_INDEX] = TRUE;
		else if (value == HIPCONFIGCALL_VAL || value == HIPCONFIGKERNEL_GPU_VAL)
			inuse[HIPCONFIGCALL_INDEX] = TRUE;
		else if (value == HIPMEMCPYASYNC_VAL || value == HIPMEMCPYASYNC_GPU_VAL)
			inuse[HIPMEMCPYASYNC_INDEX] = TRUE;
		else if (value == HIPDEVICERESET_VAL)
			inuse[HIPDEVICERESET_INDEX] = TRUE;
		else if (value == HIPTHREADEXIT_VAL)
			inuse[HIPTHREADEXIT_INDEX] = TRUE;
		else if (value == HIPSTREAMCREATE_VAL)
			inuse[HIPSTREAMCREATE_INDEX] = TRUE;
		else if (value == HIPSTREAMDESTROY_VAL)
			inuse[HIPSTREAMDESTROY_INDEX] = TRUE;
		else if (value == HIPMALLOC_VAL || value == HIPMALLOCPITCH_VAL ||
			value == HIPFREE_VAL || value == HIPMALLOCARRAY_VAL ||
			value == HIPFREEARRAY_VAL || value == HIPMALLOCHOST_VAL ||
			value == HIPFREEHOST_VAL)
			inuse[HIPMALLOC_INDEX] = TRUE;
		else if (value == HIPHOSTALLOC_VAL)
			inuse[HIPHOSTALLOC_INDEX] = TRUE;
		else if (value == HIPMEMSET_VAL)
			inuse[HIPMEMSET_INDEX] = TRUE;
		else if (value == HIPEVENTRECORD_VAL)
			inuse[HIPEVENTRECORD_INDEX] = TRUE;
		else if (value == HIPEVENTSYNCHRONIZE_VAL)
			inuse[HIPEVENTSYNCHRONIZE_INDEX] = TRUE;
		}
}


#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_HIP_Operations (void)
{
	int res, i, tmp[MAX_HIP_INDEX];

	res = MPI_Reduce (inuse, tmp, MAX_HIP_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing HIP enabled operations");

	for (i = 0; i < MAX_HIP_INDEX; i++)
		inuse[i] = tmp[i];
}

#endif

void HIPEvent_WriteEnabledOperations (FILE * fd)
{
	int anyused = FALSE;
	int i;

	for (i = 0; i < MAX_HIP_INDEX; i++)
		anyused = anyused || inuse[i];

	if (anyused)
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d    %d    HIP library call\n", 0, HIPCALL_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n");

		if (inuse[HIPLAUNCH_INDEX])
			fprintf (fd, "%d hipLaunch\n", HIPLAUNCH_VAL);

		if (inuse[HIPCONFIGCALL_INDEX])
			fprintf (fd, "%d hipConfigureCall\n", HIPCONFIGCALL_VAL);

		if (inuse[HIPMEMCPY_INDEX])
			fprintf (fd, "%d hipMemcpy\n", HIPMEMCPY_VAL);

		if (inuse[HIPTHREADBARRIER_INDEX])
			fprintf (fd, "%d hipThreadSynchronize/hipDeviceSynchronize\n", HIPTHREADBARRIER_VAL);

		if (inuse[HIPSTREAMBARRIER_INDEX])
			fprintf (fd, "%d hipStreamSynchronize\n", HIPSTREAMBARRIER_VAL);

		if (inuse[HIPMEMCPYASYNC_INDEX])
			fprintf (fd, "%d hipMemcpyAsync\n", HIPMEMCPYASYNC_VAL);

		if (inuse[HIPDEVICERESET_INDEX])
			fprintf (fd, "%d hipDeviceReset\n", HIPDEVICERESET_VAL);

		if (inuse[HIPTHREADEXIT_INDEX])
			fprintf (fd, "%d hipThreadExit\n", HIPTHREADEXIT_VAL);

		if (inuse[HIPSTREAMCREATE_INDEX])
			fprintf (fd, "%d hipStreamCreate\n", HIPSTREAMCREATE_VAL);

		if (inuse[HIPSTREAMDESTROY_INDEX])
			fprintf (fd, "%d hipStreamDestroy\n", HIPSTREAMDESTROY_VAL);

		if (inuse[HIPMALLOC_INDEX])
		{
			fprintf(fd, "%d hipMalloc\n", HIPMALLOC_VAL);
			fprintf(fd, "%d hipMallocPitch\n", HIPMALLOCPITCH_VAL);
			fprintf(fd, "%d hipFree\n", HIPFREE_VAL);
			fprintf(fd, "%d hipMallocArray\n", HIPMALLOCARRAY_VAL);
			fprintf(fd, "%d hipFreeArray\n", HIPFREEARRAY_VAL);
			fprintf(fd, "%d hipMallocHost\n", HIPMALLOCHOST_VAL);
			fprintf(fd, "%d hipFreeHost\n", HIPFREEHOST_VAL);
		}

		if (inuse[HIPHOSTALLOC_INDEX])
			fprintf(fd, "%d hipHostAlloc\n", HIPHOSTALLOC_VAL);

		if (inuse[HIPMEMSET_INDEX])
			fprintf(fd, "%d hipMemset\n", HIPMEMSET_VAL);


		if (inuse[HIPEVENTRECORD_INDEX])
		{
			fprintf(fd, "%d hipEventRecord\n", HIPEVENTRECORD_VAL);
		}

		if (inuse[HIPEVENTSYNCHRONIZE_INDEX])
		{
			fprintf(fd, "%d hipEventSynchronize\n", HIPEVENTSYNCHRONIZE_VAL);
		}


		fprintf (fd, "\n");


		if (inuse[HIPSTREAMBARRIER_INDEX])
			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    Synchronized stream (on thread)\n"
                         "\n",
                         0, HIPSTREAMBARRIER_THID_EV);

		if (inuse[HIPLAUNCH_INDEX])
		{
			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    HIP Kernel blocks per grid\n"
			             "\n",
			             0, HIP_KERNEL_BLOCKS_PER_GRID);

			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    HIP Kernel threads per block\n"
			             "\n",
			             0, HIP_KERNEL_THREADS_PER_BLOCK);
		}

		if (inuse[HIPMEMCPYASYNC_INDEX] || inuse[HIPMEMCPY_INDEX])
		{
			fprintf (fd, "EVENT_TYPE\n"
						 "%d    %d    HIP memory transfer\n", 0, HIP_MEMORY_TRANSFER);
			fprintf (fd, "VALUES\n"
						 "0 End\n");

			if (inuse[HIPMEMCPY_INDEX])
				fprintf (fd, "%d hipMemcpy\n", HIPMEMCPY_VAL);
			if (inuse[HIPMEMCPYASYNC_INDEX])
				fprintf (fd, "%d hipMemcpyAsync\n", HIPMEMCPYASYNC_VAL);
		}

		if (inuse[HIPMALLOC_INDEX] || inuse[HIPMEMCPY_INDEX] ||
		  inuse[HIPMEMCPYASYNC_INDEX] || inuse[HIPHOSTALLOC_INDEX] ||
		  inuse[HIPMEMSET_INDEX])
			fprintf (fd, "EVENT_TYPE\n"
			              "%d    %d    HIP Dynamic memory size\n"
			              "\n",
			              0, HIP_DYNAMIC_MEM_SIZE_EV);

		if (inuse[HIPMALLOC_INDEX] || inuse[HIPHOSTALLOC_INDEX] ||
		  inuse[HIPMEMSET_INDEX])
			fprintf(fd, "EVENT_TYPE\n"
			            "%d    %d    HIP Dynamic memory pointer\n"
						"\n",
						0, HIP_DYNAMIC_MEM_PTR_EV);




		if (inuse[HIPUNTRACKED_INDEX])
			fprintf(fd, "EVENT_TYPE\n"
			  "%d    %d\tHIP Untracked event\n"
			  "\n",
			  0, HIP_UNTRACKED_EV);
	}
}
