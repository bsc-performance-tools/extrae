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

#define DEFAULT_HIP_EVENTS_BLOCK_SIZE 1024

extern unsigned hip_events_block_size;

#define XTR_HIP_SET_EVENTS_BLOCK_SIZE(value) { hip_events_block_size = value;}
#define XTR_HIP_EVENTS_BLOCK_SIZE hip_events_block_size

#define XTR_FLUSH_ALL_DEVICES -1
#define XTR_FLUSH_ALL_STREAMS -1

#define HIP_SUCCESS 0

#define CHECK_HI_ERROR(err, hifunc)                             \
  if (err != HIP_SUCCESS)                                      \
    {                                                           \
      printf ("Error %d for HIP Driver API function '%s'.\n",  \
              err,  #hifunc);                                   \
      exit(-1);                                                 \
    }


/* Information per stream required during tracing */


struct RegisteredStreams_t
{
	UINT64 host_reference_time;
	gpu_event_t *device_reference_event;
	unsigned threadid; /* In Paraver sense */
	hipStream_t stream;

	gpu_event_list_t gpu_event_list;
};

struct HIPdevices_t
{
	struct RegisteredStreams_t *Stream;
	int nstreams;
	gpu_event_list_t availableEvents; /* available events to add to stream to obtain gpu timings */
	int initialized;
#if 0
	/* To perform sampling, CUPTI */
	HIcontext context;
	HIdevice device;
#endif
};

typedef struct KernelMapEntry
{
    hipFunction_t function_handle;   // Key
    char* kernel_name;               // Value
    struct KernelMapEntry* next;     // Linked list in bucket
} KernelMapEntry;

void Extrae_HIP_updateDepth_(int);
int  Extrae_HIP_getDepth();

void Extrae_HIP_flush_streams(int device_id, int stream_id);
void Extrae_hipConfigureCall_Enter(dim3, dim3, size_t, hipStream_t);
void Extrae_hipConfigureCall_Exit(void);
void Extrae_hipLaunchKernel_Enter(const char*, unsigned, unsigned, size_t, hipStream_t);
void Extrae_hipLaunchKernel_Exit(void);
void Extrae_hipModuleLaunchKernel_Enter(hipFunction_t, unsigned, unsigned, size_t, hipStream_t);
void Extrae_hipModuleLaunchKernel_Exit(void);
void Extrae_hipMalloc_Enter(unsigned int, void **, size_t);
void Extrae_hipMalloc_Exit(unsigned int);
void Extrae_hipFree_Enter(unsigned int, void *);
void Extrae_hipFree_Exit(unsigned int);
void Extrae_hipHostAlloc_Enter(void **, size_t);
void Extrae_hipHostAlloc_Exit(void);
void Extrae_hipDeviceSynchronize_Enter(void);
void Extrae_hipDeviceSynchronize_Exit(void);
void Extrae_hipThreadSynchronize_Enter(void);
void Extrae_hipThreadSynchronize_Exit(void);
void Extrae_hipEventRecord_Enter(void);
void Extrae_hipEventRecord_Exit(void);
void Extrae_hipEventSynchronize_Enter(void);
void Extrae_hipEventSynchronize_Exit(void);
void Extrae_hipStreamCreate_Enter(hipStream_t*);
void Extrae_hipStreamCreate_Exit(void);
void Extrae_hipStreamDestroy_Enter(hipStream_t);
void Extrae_hipStreamDestroy_Exit(void);
void Extrae_hipStreamSynchronize_Enter(hipStream_t);
void Extrae_hipStreamSynchronize_Exit(void);
void Extrae_hipMemcpy_Enter(void*, const void*, size_t, enum hipMemcpyKind);
void Extrae_hipMemcpy_Exit(void);
void Extrae_hipMemcpyAsync_Enter(void*, const void*, size_t, enum hipMemcpyKind, hipStream_t);
void Extrae_hipMemcpyAsync_Exit(void);
void Extrae_hipMemcpyHtoD_Enter(void*, const void*, size_t);
void Extrae_hipMemcpyHtoD_Exit(void);
void Extrae_hipMemcpyHtoDAsync_Enter(void*, const void*, size_t, hipStream_t);
void Extrae_hipMemcpyHtoDAsync_Exit(void);
void Extrae_hipMemcpyDtoH_Enter(void*, const void*, size_t);
void Extrae_hipMemcpyDtoH_Exit(void);
void Extrae_hipMemcpyDtoHAsync_Enter(void*, const void*, size_t, hipStream_t);
void Extrae_hipMemcpyDtoHAsync_Exit(void);
void Extrae_hipMemset_Enter(void*, size_t);
void Extrae_hipMemset_Exit(void);
void Extrae_hipDeviceReset_Enter(void);
void Extrae_hipDeviceReset_Exit(void);
void Extrae_hipThreadExit_Enter(void);
void Extrae_hipThreadExit_Exit(void);

void Extrae_reallocate_HIP_info(unsigned old_threads, unsigned nthreads);

void Extrae_HIP_finalize(void);

void Extrae_HIP_Initialize(int);
void Extrae_HIP_deInitialize(int);
