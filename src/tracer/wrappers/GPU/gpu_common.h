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

#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif
#include "gpu_macros.h"

#ifndef GPU_MAP_NUM_BUCKETS
#define GPU_MAP_NUM_BUCKETS 4096u
#endif

#define DEFAULT_GPU_EVENTS_BLOCK_SIZE 1024

#define XTR_FLUSH_ALL_DEVICES -1
#define XTR_FLUSH_ALL_STREAMS -1

#define SearchStream(device_id, stream) SearchAndRegisterStream(device_id, stream, FALSE)
#define RegisterStream(device_id, stream) SearchAndRegisterStream(device_id, stream, TRUE)

extern unsigned gpu_events_block_size;

#define XTR_GPU_SET_EVENTS_BLOCK_SIZE(value) { gpu_events_block_size = value;}
#define XTR_GPU_EVENTS_BLOCK_SIZE gpu_events_block_size

typedef enum {
	EXTRAE_GPU_NEW_TIME,
	EXTRAE_GPU_PREVIOUS_TIME
} Extrae_GPU_Time_Type;

typedef struct gpu_event_t gpu_event_t;

typedef struct gpu_event_t {
	GPU_EVENT_T ts_event;			 /**< HIP event timestamp. */
	unsigned event;                  /**< Event identifier. */
	unsigned long long value;        /**< Event value. */
	unsigned tag;                    /**< Event tag. */
	size_t memSize;                  /**< MemCopy size. */
	unsigned blocksPerGrid;          /**< Kernel blocks grid size. */
	unsigned threadsPerBlock;        /**< Kernel threads per block size. */
	Extrae_GPU_Time_Type timetype;  /**< HIP timing type. */
	gpu_event_t* next;               /**< Pointer to the next gpu_event_t in the list. */
} gpu_event_t;

typedef struct {
	gpu_event_t* head;
	gpu_event_t* tail;
	int autoexpand;
	size_t chunk_size;
} gpu_event_list_t;

struct RegisteredStream_t {
	UINT64 host_reference_time;
	gpu_event_t* device_reference_event;
	unsigned thread_id; /* In Paraver sense */
	GPU_STREAM_T stream;
	unsigned long long stream_id;
	gpu_event_list_t gpu_event_list;
};

struct DeviceInfo_t {
	struct RegisteredStream_t* streams;
	int num_streams;
	gpu_event_list_t available_events; /* available events to add to stream to obtain gpu timings */
	int initialized;
};

typedef struct KernelMapEntry
{
    GPU_FUNCTION_T function_handle;   // Key
    char* kernel_name;               // Value
    struct KernelMapEntry* next;     // Linked list in bucket
} KernelMapEntry;

extern pthread_mutex_t lastTagMutex;
extern pthread_rwlock_t kernel_map_rwlock;

typedef struct {
	int stream_idx;
	GPU_STREAM_T stream;
	GPU_MEMCPY_KIND_T memcpyKind;
	size_t memcpySize;
	unsigned tag;
} GPU_thread_args_t;

extern __thread GPU_thread_args_t GPU_thread_args;

unsigned GetGPUCommTag(void);
void DeinitializeDevice(int device_id);
int SearchAndRegisterStream(int device_id, GPU_STREAM_T stream, int register_stream);
void UnregisterStream(int device_id, GPU_STREAM_T stream);
void AddEventToStream(Extrae_GPU_Time_Type timetype, int device_id, int stream_idx, unsigned event, unsigned long long value, unsigned tag, size_t size, unsigned int blockspergrid, unsigned int threadsperblock);

/**************************************gpu_event_info**************************************/
void gpuEventList_init(gpu_event_list_t*, int, size_t);
void gpuEventList_allocate_chunk(gpu_event_list_t*, size_t);
void gpuEventList_add(gpu_event_list_t*, gpu_event_t*);
gpu_event_t* gpuEventList_pop(gpu_event_list_t*);
gpu_event_t* gpuEventList_peek_tail(gpu_event_list_t*);
int gpuEventList_isempty(gpu_event_list_t*);
void gpuEventList_free(gpu_event_list_t*);
/******************************************************************************************/

void FlushStreams(int, int);
void Extrae_gpuFinalize(void);
