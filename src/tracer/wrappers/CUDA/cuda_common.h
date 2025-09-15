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

#include <config.h>

#include "gpu_event_info.h"

#define DEFAULT_CUDA_EVENTS_BLOCK_SIZE 1024

extern unsigned cuda_events_block_size;

#define XTR_CUDA_SET_EVENTS_BLOCK_SIZE(value) { cuda_events_block_size = value;}
#define XTR_CUDA_EVENTS_BLOCK_SIZE cuda_events_block_size

#define XTR_FLUSH_ALL_DEVICES -1
#define XTR_FLUSH_ALL_STREAMS -1

#define CUDA_SUCCESS 0

#define CHECK_CU_ERROR(err, cufunc)                             \
  if (err != CUDA_SUCCESS)                                      \
    {                                                           \
      printf ("Error %d for CUDA Driver API function '%s'.\n",  \
              err,  #cufunc);                                   \
      exit(-1);                                                 \
    }


/* Information per stream required during tracing */


struct RegisteredStreams_t
{
	UINT64 host_reference_time;
	gpu_event_t *device_reference_event;
	unsigned threadid; /* In Paraver sense */
	cudaStream_t stream;

	gpu_event_list_t gpu_event_list;
};

struct CUDAdevices_t
{
	struct RegisteredStreams_t *Stream;
	int nstreams;
	gpu_event_list_t availableEvents; /* available events to add to stream to obtain gpu timings */
	int initialized;
#if 0
	/* To perform sampling, CUPTI */
	CUcontext context;
	CUdevice device;
#endif
};

void Extrae_CUDA_flush_streams (int device_id, int stream_id);
void Extrae_cudaConfigureCall_Enter (dim3, dim3, size_t, cudaStream_t);
void Extrae_cudaConfigureCall_Exit (void);
void Extrae_cudaLaunch_Enter (const char *f, unsigned int blocksPerGrid, unsigned int threadsPerBlock, size_t sharedMemBytes, cudaStream_t stream);
void Extrae_cudaLaunch_Exit (void);
void Extrae_cudaMalloc_Enter(unsigned int, void **, size_t);
void Extrae_cudaMalloc_Exit(unsigned int);
void Extrae_cudaFree_Enter(unsigned int, void *);
void Extrae_cudaFree_Exit(unsigned int);
void Extrae_cudaHostAlloc_Enter(void **, size_t);
void Extrae_cudaHostAlloc_Exit();
void Extrae_cudaThreadSynchronize_Enter (void);
void Extrae_cudaThreadSynchronize_Exit (void);
void Extrae_cudaDeviceSynchronize_Enter (void);
void Extrae_cudaDeviceSynchronize_Exit (void);
void Extrae_cudaEventRecord_Enter (void);
void Extrae_cudaEventRecord_Exit (void);
void Extrae_cudaEventSynchronize_Enter (void);
void Extrae_cudaEventSynchronize_Exit (void);
void Extrae_cudaStreamCreate_Enter (cudaStream_t*);
void Extrae_cudaStreamCreate_Exit (void);
void Extrae_cudaStreamDestroy_Enter (cudaStream_t);
void Extrae_cudaStreamDestroy_Exit (void);
void Extrae_cudaStreamSynchronize_Enter (cudaStream_t);
void Extrae_cudaStreamSynchronize_Exit (void);
void Extrae_cudaMemcpy_Enter (void*, const void*, size_t, enum cudaMemcpyKind);
void Extrae_cudaMemcpy_Exit (void);
void Extrae_cudaMemcpyAsync_Enter (void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);
void Extrae_cudaMemcpyAsync_Exit (void);
void Extrae_cudaMemset_Enter(void *, size_t);
void Extrae_cudaMemset_Exit();
void Extrae_cudaDeviceReset_Enter (void);
void Extrae_cudaDeviceReset_Exit (void);
void Extrae_cudaThreadExit_Enter (void);
void Extrae_cudaThreadExit_Exit (void);

void Extrae_reallocate_CUDA_info (unsigned old_threads, unsigned nthreads);

void Extrae_CUDA_finalize (void);

void Extrae_CUDA_Initialize(int);
void Extrae_CUDA_deInitialize(int);
