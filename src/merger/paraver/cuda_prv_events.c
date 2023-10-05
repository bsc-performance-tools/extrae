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
#ifdef HAVE_BFD
# include "addr2info.h"
#endif

#include "events.h"
#include "mpi2out.h"
#include "options.h"

#define CUDALAUNCH_INDEX           0
#define CUDACONFIGCALL_INDEX       1
#define CUDAMEMCPY_INDEX           2
#define CUDATHREADBARRIER_INDEX    3
#define CUDASTREAMBARRIER_INDEX    4
#define CUDAMEMCPYASYNC_INDEX      5
#define CUDATHREADEXIT_INDEX       6
#define CUDADEVICERESET_INDEX      7
#define CUDASTREAMCREATE_INDEX     8
#define CUDASTREAMDESTROY_INDEX    9
#define CUDAMALLOC_INDEX          10
#define CUDAHOSTALLOC_INDEX       11
#define CUDAMEMSET_INDEX          12
#define CUDAUNTRACKED_INDEX       13

#define MAX_CUDA_INDEX            14

static int inuse[MAX_CUDA_INDEX] = { FALSE };

void Enable_CUDA_Operation (int type)
{
	if (type == CUDALAUNCH_VAL || type == CUDAKERNEL_GPU_VAL)
		inuse[CUDALAUNCH_INDEX] = TRUE;
	else if (type == CUDAMEMCPY_VAL || type == CUDAMEMCPY_GPU_VAL)
		inuse[CUDAMEMCPY_INDEX] = TRUE;
	else if (type == CUDASTREAMBARRIER_VAL)
		inuse[CUDASTREAMBARRIER_INDEX] = TRUE;
	else if (type == CUDATHREADBARRIER_VAL || type == CUDATHREADBARRIER_GPU_VAL)
		inuse[CUDATHREADBARRIER_INDEX] = TRUE;
	else if (type == CUDACONFIGCALL_VAL || type == CUDACONFIGKERNEL_GPU_VAL)
		inuse[CUDACONFIGCALL_INDEX] = TRUE;
	else if (type == CUDAMEMCPYASYNC_VAL || type == CUDAMEMCPYASYNC_GPU_VAL)
		inuse[CUDAMEMCPYASYNC_INDEX] = TRUE;
	else if (type == CUDADEVICERESET_VAL)
		inuse[CUDADEVICERESET_INDEX] = TRUE;
	else if (type == CUDATHREADEXIT_VAL)
		inuse[CUDATHREADEXIT_INDEX] = TRUE;
	else if (type == CUDASTREAMCREATE_VAL)
		inuse[CUDASTREAMCREATE_INDEX] = TRUE;
	else if (type == CUDASTREAMDESTROY_VAL)
		inuse[CUDASTREAMDESTROY_INDEX] = TRUE;
	else if (type == CUDAMALLOC_VAL || type == CUDAMALLOCPITCH_VAL ||
	  type == CUDAFREE_VAL || type == CUDAMALLOCARRAY_VAL ||
	  type == CUDAFREEARRAY_VAL || type == CUDAMALLOCHOST_VAL ||
	  type == CUDAFREEHOST_VAL)
		inuse[CUDAMALLOC_INDEX] = TRUE;
	else if (type == CUDAHOSTALLOC_VAL)
		inuse[CUDAHOSTALLOC_INDEX] = TRUE;
	else if (type == CUDAMEMSET_VAL)
		inuse[CUDAMEMSET_INDEX] = TRUE;
	else if (type == CUDA_UNTRACKED_EV)
		inuse[CUDAUNTRACKED_INDEX] = TRUE;
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_CUDA_Operations (void)
{
	int res, i, tmp[MAX_CUDA_INDEX];

	res = MPI_Reduce (inuse, tmp, MAX_CUDA_INDEX, MPI_INT, MPI_BOR, 0,
		MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "While sharing CUDA enabled operations");

	for (i = 0; i < MAX_CUDA_INDEX; i++)
		inuse[i] = tmp[i];
}

#endif

void CUDAEvent_WriteEnabledOperations (FILE * fd)
{
	int anyused = FALSE;
	int i;

	for (i = 0; i < MAX_CUDA_INDEX; i++)
		anyused = anyused || inuse[i];

	if (anyused)
	{
		fprintf (fd, "EVENT_TYPE\n"
		             "%d   %d    CUDA library call\n", 0, CUDACALL_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n");

		if (inuse[CUDALAUNCH_INDEX])
			fprintf (fd, "%d cudaLaunch\n", CUDALAUNCH_VAL);

		if (inuse[CUDACONFIGCALL_INDEX])
			fprintf (fd, "%d cudaConfigureCall\n", CUDACONFIGCALL_VAL);

		if (inuse[CUDAMEMCPY_INDEX])
			fprintf (fd, "%d cudaMemcpy\n", CUDAMEMCPY_VAL);

		if (inuse[CUDATHREADBARRIER_INDEX])
			fprintf (fd, "%d cudaThreadSynchronize/cudaDeviceSynchronize\n", CUDATHREADBARRIER_VAL);

		if (inuse[CUDASTREAMBARRIER_INDEX])
			fprintf (fd, "%d cudaStreamSynchronize\n", CUDASTREAMBARRIER_VAL);

		if (inuse[CUDAMEMCPYASYNC_INDEX])
			fprintf (fd, "%d cudaMemcpyAsync\n", CUDAMEMCPYASYNC_VAL);

		if (inuse[CUDADEVICERESET_INDEX])
			fprintf (fd, "%d cudaDeviceReset\n", CUDADEVICERESET_VAL);

		if (inuse[CUDATHREADEXIT_INDEX])
			fprintf (fd, "%d cudaThreadExit\n", CUDATHREADEXIT_VAL);

		if (inuse[CUDASTREAMCREATE_INDEX])
			fprintf (fd, "%d cudaStreamCreate\n", CUDASTREAMCREATE_VAL);

		if (inuse[CUDASTREAMDESTROY_INDEX])
			fprintf (fd, "%d cudaStreamDestroy\n", CUDASTREAMDESTROY_VAL);

		if (inuse[CUDAMALLOC_INDEX])
		{
			fprintf(fd, "%d cudaMalloc\n", CUDAMALLOC_VAL);
			fprintf(fd, "%d cudaMallocPitch\n", CUDAMALLOCPITCH_VAL);
			fprintf(fd, "%d cudaFree\n", CUDAFREE_VAL);
			fprintf(fd, "%d cudaMallocArray\n", CUDAMALLOCARRAY_VAL);
			fprintf(fd, "%d cudaFreeArray\n", CUDAFREEARRAY_VAL);
			fprintf(fd, "%d cudaMallocHost\n", CUDAMALLOCHOST_VAL);
			fprintf(fd, "%d cudaFreeHost\n", CUDAFREEHOST_VAL);
		}

		if (inuse[CUDAHOSTALLOC_INDEX])
			fprintf(fd, "%d cudaHostAlloc\n", CUDAHOSTALLOC_VAL);

		if (inuse[CUDAMEMSET_INDEX])
			fprintf(fd, "%d cudaMemset\n", CUDAMEMSET_VAL);

		fprintf (fd, "\n");

		if (inuse[CUDAMALLOC_INDEX] || inuse[CUDAMEMCPY_INDEX] ||
		  inuse[CUDAMEMCPYASYNC_INDEX] || inuse[CUDAHOSTALLOC_INDEX] ||
		  inuse[CUDAMEMSET_INDEX])
			fprintf (fd, "EVENT_TYPE\n"
			              "%d   %d    CUDA Dynamic memory size\n"
			              "\n",
			              0, CUDA_DYNAMIC_MEM_SIZE_EV);

		if (inuse[CUDAMALLOC_INDEX] || inuse[CUDAHOSTALLOC_INDEX] ||
		  inuse[CUDAMEMSET_INDEX])
			fprintf(fd, "EVENT_TYPE\n"
			            "%d   %d    CUDA Dynamic memory pointer\n"
						"\n",
						0, CUDA_DYNAMIC_MEM_PTR_EV);

		if (inuse[CUDASTREAMBARRIER_INDEX])
			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    Synchronized stream (on thread)\n"
                         "\n",
                         0, CUDASTREAMBARRIER_THID_EV);

		if (inuse[CUDAUNTRACKED_INDEX])
			fprintf(fd, "EVENT_TYPE\n"
			  "%d\t%d\tCUDA Untracked event\n"
			  "\n",
			  0, CUDA_UNTRACKED_EV);
	}
}
