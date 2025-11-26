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
	CUDALAUNCH_INDEX,
	CUDACONFIGCALL_INDEX,
	CUDAMEMCPY_INDEX,
	CUDAMEMCPYTOSYMBOL_INDEX,
	CUDAMEMCPYFROMSYMBOL_INDEX,
	CUDATHREADBARRIER_INDEX,
	CUDASTREAMSYNCHRONIZE_INDEX,
	CUDAMEMCPYASYNC_INDEX,
	CUDATHREADEXIT_INDEX,
	CUDADEVICERESET_INDEX,
	CUDASTREAMCREATE_INDEX,
	CUDASTREAMDESTROY_INDEX,
	CUDAMALLOC_INDEX,
	CUDAHOSTALLOC_INDEX,
	CUDAMEMSET_INDEX,
	CUDAMEMSETASYNC_INDEX,
	CUDAEVENTRECORD_INDEX,
	CUDAEVENTSYNCHRONIZE_INDEX,
	CUDASTREAMWAITEVENT_INDEX,
	CUDAUNTRACKED_INDEX,
	MAX_CUDA_INDEX
};


static int inuse[MAX_CUDA_INDEX] = { FALSE };

void Enable_CUDA_Operation (INT32 type, UINT64 value)
{
	if (type == CUDA_UNTRACKED_EV){
			inuse[CUDAUNTRACKED_INDEX] = TRUE;
		}
	else if(type == CUDACALL_EV || type == CUDACALLGPU_EV){
		if (value == CUDALAUNCH_VAL || value == CUDAKERNEL_GPU_VAL)
			inuse[CUDALAUNCH_INDEX] = TRUE;
		else if (value == CUDAMEMCPY_VAL || value == CUDAMEMCPY_GPU_VAL)
			inuse[CUDAMEMCPY_INDEX] = TRUE;
		else if (value == CUDAMEMCPYTOSYMBOL_VAL || value == CUDAMEMCPY_GPU_VAL)
			inuse[CUDAMEMCPYTOSYMBOL_INDEX] = TRUE;
		else if (value == CUDAMEMCPYFROMSYMBOL_VAL || value == CUDAMEMCPY_GPU_VAL)
			inuse[CUDAMEMCPYFROMSYMBOL_INDEX] = TRUE;
		else if (value == CUDASTREAMSYNCHRONIZE_VAL)
			inuse[CUDASTREAMSYNCHRONIZE_INDEX] = TRUE;
		else if (value == CUDATHREADBARRIER_VAL)
			inuse[CUDATHREADBARRIER_INDEX] = TRUE;
		else if (value == CUDACONFIGCALL_VAL || value == CUDACONFIGKERNEL_GPU_VAL)
			inuse[CUDACONFIGCALL_INDEX] = TRUE;
		else if (value == CUDAMEMCPYASYNC_VAL || value == CUDAMEMCPYASYNC_GPU_VAL)
			inuse[CUDAMEMCPYASYNC_INDEX] = TRUE;
		else if (value == CUDADEVICERESET_VAL)
			inuse[CUDADEVICERESET_INDEX] = TRUE;
		else if (value == CUDATHREADEXIT_VAL)
			inuse[CUDATHREADEXIT_INDEX] = TRUE;
		else if (value == CUDASTREAMCREATE_VAL)
			inuse[CUDASTREAMCREATE_INDEX] = TRUE;
		else if (value == CUDASTREAMDESTROY_VAL)
			inuse[CUDASTREAMDESTROY_INDEX] = TRUE;
		else if (value == CUDAMALLOC_VAL || value == CUDAMALLOCPITCH_VAL ||
			value == CUDAFREE_VAL || value == CUDAMALLOCARRAY_VAL ||
			value == CUDAFREEARRAY_VAL || value == CUDAMALLOCHOST_VAL ||
			value == CUDAFREEHOST_VAL)
			inuse[CUDAMALLOC_INDEX] = TRUE;
		else if (value == CUDAHOSTALLOC_VAL)
			inuse[CUDAHOSTALLOC_INDEX] = TRUE;
		else if (value == CUDAMEMSET_VAL)
			inuse[CUDAMEMSET_INDEX] = TRUE;
		else if (value == CUDAMEMSETASYNC_VAL)
			inuse[CUDAMEMSETASYNC_INDEX] = TRUE;
		else if (value == CUDAEVENTRECORD_VAL)
			inuse[CUDAEVENTRECORD_INDEX] = TRUE;
		else if (value == CUDAEVENTSYNCHRONIZE_VAL)
			inuse[CUDAEVENTSYNCHRONIZE_INDEX] = TRUE;
		else if (value == CUDASTREAMWAITEVENT_VAL)
			inuse[CUDASTREAMWAITEVENT_INDEX] = TRUE;
		}
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
		             "%d    %d    CUDA library call\n", 0, CUDACALL_EV);
		fprintf (fd, "VALUES\n"
		             "0 End\n");

		if (inuse[CUDALAUNCH_INDEX])
			fprintf (fd, "%d cudaLaunch\n", CUDALAUNCH_VAL);

		if (inuse[CUDACONFIGCALL_INDEX])
			fprintf (fd, "%d cudaConfigureCall\n", CUDACONFIGCALL_VAL);

		if (inuse[CUDAMEMCPY_INDEX])
			fprintf (fd, "%d cudaMemcpy\n", CUDAMEMCPY_VAL);

		if (inuse[CUDATHREADBARRIER_INDEX])
			fprintf (fd, "%d cudaDeviceSynchronize\n", CUDATHREADBARRIER_VAL);

		if (inuse[CUDASTREAMSYNCHRONIZE_INDEX])
			fprintf (fd, "%d cudaStreamSynchronize\n", CUDASTREAMSYNCHRONIZE_VAL);

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

		if (inuse[CUDAMEMCPYTOSYMBOL_INDEX])
			fprintf (fd, "%d cudaMemcpyToSymbol\n", CUDAMEMCPYTOSYMBOL_VAL);

		if (inuse[CUDAMEMCPYFROMSYMBOL_INDEX])
			fprintf (fd, "%d cudaMemcpyFromSymbol\n", CUDAMEMCPYFROMSYMBOL_VAL);

		if (inuse[CUDAMEMSET_INDEX])
			fprintf(fd, "%d cudaMemset\n", CUDAMEMSET_VAL);

		if (inuse[CUDAMEMSETASYNC_INDEX])
			fprintf(fd, "%d cudaMemsetAsync\n", CUDAMEMSETASYNC_VAL);

		if (inuse[CUDAEVENTRECORD_INDEX])
		{
			fprintf(fd, "%d cudaEventRecord\n", CUDAEVENTRECORD_VAL);
		}

		if (inuse[CUDAEVENTSYNCHRONIZE_INDEX])
		{
			fprintf(fd, "%d cudaEventSynchronize\n", CUDAEVENTSYNCHRONIZE_VAL);
		}

		if (inuse[CUDASTREAMWAITEVENT_INDEX])
		{
			fprintf(fd, "%d cudaStreamWaitEvent\n", CUDASTREAMWAITEVENT_VAL);
		}

		fprintf (fd, "\n");

		if (inuse[CUDASTREAMSYNCHRONIZE_INDEX] || inuse[CUDAEVENTRECORD_INDEX])
			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    Stream ID destination\n"
                         "\n",
                         0, CUDA_STREAM_DEST_ID_EV);

		if (inuse[CUDALAUNCH_INDEX])
		{
			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    CUDA Kernel blocks per grid\n"
			             "\n",
			             0, CUDA_KERNEL_BLOCKS_PER_GRID);

			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    CUDA Kernel threads per block\n"
			             "\n",
			             0, CUDA_KERNEL_THREADS_PER_BLOCK);
		}

		if (inuse[CUDAMEMCPYASYNC_INDEX] || inuse[CUDAMEMCPY_INDEX] || 
			inuse[CUDAMEMCPYTOSYMBOL_INDEX] || inuse[CUDAMEMCPYFROMSYMBOL_INDEX])
		{
			fprintf (fd, "EVENT_TYPE\n"
						 "%d    %d    CUDA memory transfer\n", 0, CUDA_MEMORY_TRANSFER);
			fprintf (fd, "VALUES\n"
						 "0 End\n");

			if (inuse[CUDAMEMCPY_INDEX])
				fprintf (fd, "%d cudaMemcpy\n", CUDAMEMCPY_GPU_VAL);
			if (inuse[CUDAMEMCPYASYNC_INDEX])
				fprintf (fd, "%d cudaMemcpyAsync\n", CUDAMEMCPYASYNC_GPU_VAL);
			if (inuse[CUDAMEMCPYTOSYMBOL_INDEX])
				fprintf (fd, "%d cudaMemcpyToSymbol\n", CUDAMEMCPYTOSYMBOL_GPU_VAL);
			if (inuse[CUDAMEMCPYFROMSYMBOL_INDEX])
				fprintf (fd, "%d cudaMemcpyFromSymbol\n", CUDAMEMCPYFROMSYMBOL_GPU_VAL);
		}

		if (inuse[CUDAMALLOC_INDEX] || inuse[CUDAMEMCPY_INDEX] ||
		  inuse[CUDAMEMCPYASYNC_INDEX] || inuse[CUDAHOSTALLOC_INDEX] ||
		  inuse[CUDAMEMSET_INDEX] || inuse[CUDAMEMSETASYNC_INDEX] ||
		  inuse[CUDAMEMCPYTOSYMBOL_INDEX] || inuse[CUDAMEMCPYFROMSYMBOL_INDEX])
			fprintf (fd, "EVENT_TYPE\n"
			              "%d    %d    CUDA Dynamic memory size\n"
			              "\n",
			              0, CUDA_DYNAMIC_MEM_SIZE_EV);

		if (inuse[CUDAMALLOC_INDEX] || inuse[CUDAHOSTALLOC_INDEX] ||
		  inuse[CUDAMEMSET_INDEX] || inuse[CUDAMEMSETASYNC_INDEX])
			fprintf(fd, "EVENT_TYPE\n"
			            "%d    %d    CUDA Dynamic memory pointer\n"
						"\n",
						0, CUDA_DYNAMIC_MEM_PTR_EV);

		if (inuse[CUDAEVENTSYNCHRONIZE_INDEX] || inuse[CUDAEVENTRECORD_INDEX])
			fprintf (fd, "EVENT_TYPE\n"
			             "%d    %d    cudaEvent ID\n"
                         "\n",
                         0, CUDAEVENT_ID_EV);

		if (inuse[CUDAUNTRACKED_INDEX])
			fprintf(fd, "EVENT_TYPE\n"
			  "%d    %d\tCUDA Untracked event\n"
			  "\n",
			  0, CUDA_UNTRACKED_EV);
	}
}
