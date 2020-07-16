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
#define CUDAUNKNOWN_INDEX         13

#define MAX_CUDA_INDEX            14

static int inuse[MAX_CUDA_INDEX] = { FALSE };

void Enable_CUDA_Operation (int type)
{
	if (type == CUDALAUNCH_EV || type == CUDAKERNEL_GPU_EV)
		inuse[CUDALAUNCH_INDEX] = TRUE;
	else if (type == CUDAMEMCPY_EV || type == CUDAMEMCPY_GPU_EV)
		inuse[CUDAMEMCPY_INDEX] = TRUE;
	else if (type == CUDASTREAMBARRIER_EV)
		inuse[CUDASTREAMBARRIER_INDEX] = TRUE;
	else if (type == CUDATHREADBARRIER_EV || type == CUDATHREADBARRIER_GPU_EV)
		inuse[CUDATHREADBARRIER_INDEX] = TRUE;
	else if (type == CUDACONFIGCALL_EV || type == CUDACONFIGKERNEL_GPU_EV)
		inuse[CUDACONFIGCALL_INDEX] = TRUE;
	else if (type == CUDAMEMCPYASYNC_EV || type == CUDAMEMCPYASYNC_GPU_EV)
		inuse[CUDAMEMCPYASYNC_INDEX] = TRUE;
	else if (type == CUDADEVICERESET_EV)
		inuse[CUDADEVICERESET_INDEX] = TRUE;
	else if (type == CUDATHREADEXIT_EV)
		inuse[CUDATHREADEXIT_INDEX] = TRUE;
	else if (type == CUDASTREAMCREATE_EV)
		inuse[CUDASTREAMCREATE_INDEX] = TRUE;
	else if (type == CUDASTREAMDESTROY_EV)
		inuse[CUDASTREAMDESTROY_INDEX] = TRUE;
	else if (type == CUDAMALLOC_EV || type == CUDAMALLOCPITCH_EV ||
	  type == CUDAFREE_EV || type == CUDAMALLOCARRAY_EV ||
	  type == CUDAFREEARRAY_EV || type == CUDAMALLOCHOST_EV ||
	  type == CUDAFREEHOST_EV)
		inuse[CUDAMALLOC_INDEX] = TRUE;
	else if (type == CUDAHOSTALLOC_EV)
		inuse[CUDAHOSTALLOC_INDEX] = TRUE;
	else if (type == CUDAMEMSET_EV)
		inuse[CUDAMEMSET_INDEX] = TRUE;
	else if (type == CUDAUNKNOWN_EV)
		inuse[CUDAUNKNOWN_INDEX] = TRUE;
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
			fprintf (fd, "%d cudaLaunch\n", CUDALAUNCH_EV - CUDABASE_EV);

		if (inuse[CUDACONFIGCALL_INDEX])
			fprintf (fd, "%d cudaConfigureCall\n", CUDACONFIGCALL_EV - CUDABASE_EV);

		if (inuse[CUDAMEMCPY_INDEX])
			fprintf (fd, "%d cudaMemcpy\n", CUDAMEMCPY_EV - CUDABASE_EV);

		if (inuse[CUDATHREADBARRIER_INDEX])
			fprintf (fd, "%d cudaThreadSynchronize/cudaDeviceSynchronize\n", CUDATHREADBARRIER_EV - CUDABASE_EV);

		if (inuse[CUDASTREAMBARRIER_INDEX])
			fprintf (fd, "%d cudaStreamSynchronize\n", CUDASTREAMBARRIER_EV - CUDABASE_EV);

		if (inuse[CUDAMEMCPYASYNC_INDEX])
			fprintf (fd, "%d cudaMemcpyAsync\n", CUDAMEMCPYASYNC_EV - CUDABASE_EV);

		if (inuse[CUDADEVICERESET_INDEX])
			fprintf (fd, "%d cudaDeviceReset\n", CUDADEVICERESET_EV - CUDABASE_EV);

		if (inuse[CUDATHREADEXIT_INDEX])
			fprintf (fd, "%d cudaThreadExit\n", CUDATHREADEXIT_EV - CUDABASE_EV);

		if (inuse[CUDASTREAMCREATE_INDEX])
			fprintf (fd, "%d cudaStreamCreate\n", CUDASTREAMCREATE_EV - CUDABASE_EV);

		if (inuse[CUDASTREAMDESTROY_INDEX])
			fprintf (fd, "%d cudaStreamDestroy\n", CUDASTREAMDESTROY_EV - CUDABASE_EV);

		if (inuse[CUDAMALLOC_INDEX])
		{
			fprintf(fd, "%d cudaMalloc\n", CUDAMALLOC_EV - CUDABASE_EV);
			fprintf(fd, "%d cudaMallocPitch\n", CUDAMALLOCPITCH_EV - CUDABASE_EV);
			fprintf(fd, "%d cudaFree\n", CUDAFREE_EV - CUDABASE_EV);
			fprintf(fd, "%d cudaMallocArray\n", CUDAMALLOCARRAY_EV - CUDABASE_EV);
			fprintf(fd, "%d cudaFreeArray\n", CUDAFREEARRAY_EV - CUDABASE_EV);
			fprintf(fd, "%d cudaMallocHost\n", CUDAMALLOCHOST_EV - CUDABASE_EV);
			fprintf(fd, "%d cudaFreeHost\n", CUDAFREEHOST_EV - CUDABASE_EV);
		}

		if (inuse[CUDAHOSTALLOC_INDEX])
			fprintf(fd, "%d cudaHostAlloc\n", CUDAHOSTALLOC_EV - CUDABASE_EV);

		if (inuse[CUDAMEMSET_INDEX])
			fprintf(fd, "%d cudaMemset\n", CUDAMEMSET_EV - CUDABASE_EV);

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

		if (inuse[CUDAUNKNOWN_INDEX])
			fprintf(fd, "EVENT_TYPE\n"
			  "%d\t%d\tCUDA Unknown event\n"
			  "\n",
			  0, CUDAUNKNOWN_EV);
	}
}
