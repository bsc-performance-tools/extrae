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

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "wrapper.h"
#include "taskid.h"
#include "threadinfo.h"
#include "trace_mode.h"
#include "xalloc.h"
#include "common.h"
#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include "gpu_macros.h"
#include "gpu_common.h"
#include "gpu_probe.h"

#if defined(__APPLE__)
#define HOST_NAME_MAX 512
#endif

/**
 * Variable to control the initialization and allow the flushing of
 * events at the exit point.
 */
int gpuInitialized = 0;

unsigned gpu_events_block_size = DEFAULT_GPU_EVENTS_BLOCK_SIZE;

__thread GPU_thread_args_t GPU_thread_args;

static unsigned lastTag = 0xC0DA; /* Fixed tag */

/* Array of hash buckets for the kernel map.
 * Each bucket is the head of a linked list of KernelMapEntry nodes. */
KernelMapEntry *kernel_map_buckets[GPU_MAP_NUM_BUCKETS] = {0};

pthread_mutex_t lastTagMutex = PTHREAD_MUTEX_INITIALIZER;

// Read–write lock protecting the entire kernel map.
// Readers can access concurrently, writers get exclusive access.
pthread_rwlock_t kernel_map_rwlock = PTHREAD_RWLOCK_INITIALIZER;

static struct DeviceInfo_t *deviceArray = NULL;
static int deviceCount = 0;

/**
 * @brief Initializes an event list setting the first and last elements to NULL
 *
 * @param list        Pointer to the gpu_event_list_t to initialize.
 * @param autoexpand  Boolean to allow allocation of more elements if empty.
 * @param chunk_size  Size of each allocated memory chunk.
 */
void gpuEventList_init(gpu_event_list_t *list, int autoexpand, size_t chunk_size)
{
	list->head = NULL;
	list->tail = NULL;
	list->autoexpand = autoexpand;
	list->chunk_size = chunk_size;
}

/**
 * @brief Allocates and initializes new event elements for the event list.
 *
 * Allocates `size` new gpu_event_t elements, initializes them (creating a GPU
 * event with default flags for each), and appends them to `list`.
 *
 * @param list  Pointer to the gpu_event_list_t where elements will be added.
 * @param size  Number of new event elements to allocate.
 */
void gpuEventList_allocate_chunk(gpu_event_list_t *list, size_t size)
{
	size_t i;
	gpu_event_t *event_info;

	for (i = 0; i < size; i++)
	{
		event_info = xmalloc_and_zero(sizeof(gpu_event_t));
		GPU_RUNTIME_CHECK(GPU_EVENT_CREATE_WITH_FLAGS(&(event_info->ts_event), 0));
		gpuEventList_add(list, event_info);
	}
}

/**
 * @brief Appends a gpu_event_t element to the tail of the list.
 *
 * @param list     Pointer to the gpu_event_list_t.
 * @param element  Pointer to the gpu_event_t to append.
 */
void gpuEventList_add(gpu_event_list_t *list, gpu_event_t *element)
{
	element->next = NULL;
	if (list->tail != NULL)
		list->tail->next = element;
	else
		list->head = element;
	list->tail = element;
}

/**
 * @brief Removes and returns the head element of the list.
 *
 * If autoexpand is set and the list is empty, a new chunk is allocated first.
 *
 * @param list  Pointer to the gpu_event_list_t.
 * @return      Pointer to the removed gpu_event_t, or NULL if list was empty.
 */
gpu_event_t* gpuEventList_pop(gpu_event_list_t *list)
{
	if (list->autoexpand && gpuEventList_isempty(list))
		gpuEventList_allocate_chunk(list, list->chunk_size);

	gpu_event_t *element = list->head;
	if (element != NULL)
	{
		list->head = element->next;
		if (list->tail == element)
			list->tail = NULL;
		element->next = NULL;
	}
	return element;
}

/**
 * @brief Returns the tail element without removing it.
 *
 * @param list  Pointer to the gpu_event_list_t.
 * @return      Pointer to the tail gpu_event_t, or NULL if the list is empty.
 */
gpu_event_t* gpuEventList_peek_tail(gpu_event_list_t *list)
{
	return list->tail;
}

/**
 * @brief Returns 1 if the list is empty, 0 otherwise.
 */
int gpuEventList_isempty(gpu_event_list_t *list)
{
	return list->head == NULL;
}

/**
 * @brief Frees all elements in the list and destroys their associated GPU events.
 *
 * @param list  Pointer to the gpu_event_list_t to free.
 */
void gpuEventList_free(gpu_event_list_t *list)
{
	if (list == NULL)
		return;

	gpu_event_t *current = list->head;
	gpu_event_t *next;

	while (current)
	{
		next = current->next;
		GPU_RUNTIME_CHECK(GPU_EVENT_DESTROY(current->ts_event));
		xfree(current);
		current = next;
	}
	list->head = NULL;
	list->tail = NULL;
}

unsigned GetGPUCommTag(void)
{
	unsigned new_tag = 0;
	pthread_mutex_lock(&lastTagMutex);
	new_tag = lastTag++;
	pthread_mutex_unlock(&lastTagMutex);
	return new_tag;
}

static void SynchronizeStream(int device_id, int stream_idx)
{
	if (device_id >= deviceCount)
	{
		fprintf(stderr, PACKAGE_NAME ": Error! Invalid GPU device id in SynchronizeStream\n");
		exit(-1);
	}

#ifdef DEBUG
	fprintf(stderr, "SynchronizeStream (device_id=%d, stream_idx=%d, stream=%p)\n",
			device_id, stream_idx,
			deviceArray[device_id].streams[stream_idx].stream);
#endif

	if (deviceArray[device_id].streams[stream_idx].device_reference_event == NULL)
		deviceArray[device_id].streams[stream_idx].device_reference_event = gpuEventList_pop(&deviceArray[device_id].available_events);

	GPU_RUNTIME_CHECK(GPU_EVENT_RECORD(deviceArray[device_id].streams[stream_idx].device_reference_event->ts_event, deviceArray[device_id].streams[stream_idx].stream));
	GPU_RUNTIME_CHECK(GPU_EVENT_SYNCHRONIZE(deviceArray[device_id].streams[stream_idx].device_reference_event->ts_event));

	deviceArray[device_id].streams[stream_idx].host_reference_time = TIME;
}

void DeinitializeDevice(int device_id)
{
	if (deviceArray != NULL)
	{
		if (deviceArray[device_id].initialized)
		{
			xfree(deviceArray[device_id].streams);
			deviceArray[device_id].streams = NULL;
			deviceArray[device_id].initialized = FALSE;
		}
	}
}

/* ══════════════════════════════════════════════════════════════════════════
 * SearchAndRegisterStream
 *
 * NOTE: This function intentionally keeps two separate #ifdef blocks for
 * stream identity lookup because HIP and CUDA use fundamentally different
 * mechanisms:
 *
 *   - CUDA: cudaStreamGetId() returns a stable 64-bit unique identifier
 *           that survives pointer reuse after stream destruction.
 *   - HIP:  No equivalent API exists (as of ROCm ≤ 7.0). Identity is
 *           determined by comparing the raw stream pointer directly.
 *
 * There is no common abstraction that preserves this semantic difference,
 * so the #ifdef is correct here and should NOT be removed.
 * ══════════════════════════════════════════════════════════════════════════ */
int SearchAndRegisterStream(int device_id, GPU_STREAM_T stream, int register_stream)
{
	int stream_idx = 0;
	unsigned long long unique_stream_id = 0;
	int i = 0;

	/* Lazy-initialize the device array on first call */
	if (deviceArray == NULL)
	{
		GPU_RUNTIME_CHECK(GPU_GET_DEVICE_COUNT(&deviceCount));

		deviceArray = (struct DeviceInfo_t *)xmalloc(sizeof(struct DeviceInfo_t) * deviceCount);

		for (i = 0; i < deviceCount; i++)
		{
			deviceArray[i].initialized = FALSE;
			deviceArray[i].streams = NULL;
			deviceArray[i].num_streams = 0;
		}

		atexit(Extrae_gpuFinalize);
		gpuInitialized = 1;
	}

	/* Per-device initialization on first use */
	if (!deviceArray[device_id].initialized)
	{
		gpuEventList_init(&deviceArray[device_id].available_events, TRUE, XTR_GPU_EVENTS_BLOCK_SIZE);
		deviceArray[device_id].initialized = TRUE;
	}

	struct RegisteredStream_t *StreamArray = deviceArray[device_id].streams;

#if defined(CUDA_SUPPORT)
	/*
	 * CUDA: obtain a stable numeric stream ID and search by value.
	 * cudaStreamGetId() was introduced in CUDA 12.0 and is the only
	 * reliable way to detect stream aliasing after destroy+recreate.
	 */
	GPU_RUNTIME_CHECK(cudaStreamGetId(stream, &unique_stream_id));
	for (stream_idx = 0; stream_idx < deviceArray[device_id].num_streams; stream_idx++)
	{
		if (StreamArray[stream_idx].stream_id == unique_stream_id)
			return stream_idx;
	}
#endif
#if defined(HIP_SUPPORT)
	/*
	 * HIP: no hipStreamGetId() equivalent exists (ROCm ≤ 7.0 / activity API
	 * caveat — revisit when ROCm 7.1+ ships). Fall back to pointer comparison.
	 */
	for (stream_idx = 0; stream_idx < deviceArray[device_id].num_streams; stream_idx++)
	{
		if (StreamArray[stream_idx].stream == stream)
			return stream_idx;
	}
#endif

	/* Stream not found — register only if requested */
	if (!register_stream)
		return -1;

	Probe_Gpu_StreamRegister_Entry();

	/* Grow the stream array by one slot */
	StreamArray = deviceArray[device_id].streams = (struct RegisteredStream_t *)xrealloc(StreamArray, (stream_idx + 1) * sizeof(struct RegisteredStream_t));

	stream_idx = deviceArray[device_id].num_streams;
	deviceArray[device_id].num_streams++;

	/*
	 * XXX Should this be Backend_getMaximumOfThreads()?  If the thread count
	 * was previously increased and then decreased, we may end up with a line
	 * carrying mixed thread+stream semantics.
	 */
	Backend_ChangeNumberOfThreads(Backend_getNumberOfThreads() + 1);

	StreamArray[stream_idx].host_reference_time = 0;
	StreamArray[stream_idx].device_reference_event = NULL;
	StreamArray[stream_idx].thread_id = Backend_getNumberOfThreads() - 1;
	StreamArray[stream_idx].stream = stream;
	StreamArray[stream_idx].stream_id = unique_stream_id;
	gpuEventList_init(&StreamArray[stream_idx].gpu_event_list, FALSE, XTR_GPU_EVENTS_BLOCK_SIZE);

#ifdef DEBUG
	fprintf(stderr, "SearchAndRegisterStream(device_id=%d, stream=%p) => stream_idx=%d\n",
			device_id, stream, stream_idx);
#endif

	/* Assign a descriptive name to the thread bound to this stream */
	{
		char _threadname[THREAD_INFO_NAME_LEN];
		char _hostname[HOST_NAME_MAX];

		if (gethostname(_hostname, HOST_NAME_MAX) == 0)
			sprintf(_threadname, "GPU-D%d.S%d-%s", device_id + 1, stream_idx + 1, _hostname);
		else
			sprintf(_threadname, "GPU-D%d.S%d-%s", device_id + 1, stream_idx + 1, "unknown-host");

		Extrae_set_thread_name(StreamArray[stream_idx].thread_id, _threadname);
	}

	/* Record a reference timestamp on the new stream */
	/* FIX: CU_EVENT_BLOCKING_SYNC may be harmful — keep under review */
	StreamArray[stream_idx].device_reference_event = NULL;
	SynchronizeStream(device_id, stream_idx);

	/*
	 * Transition the GPU stream base state from NOT_TRACING to IDLE.
	 * Emitted manually because GPU burst mode is not yet supported.
	 */
	THREAD_TRACE_MISCEVENT(StreamArray[stream_idx].thread_id, StreamArray[stream_idx].host_reference_time, TRACING_MODE_EV, TRACE_MODE_DETAIL, 0);

	Probe_Gpu_StreamRegister_Exit();

	return stream_idx;
}

/* ══════════════════════════════════════════════════════════════════════════
 * traceGPUEvents
 *
 * Emits GPU events into the tracing buffer and inserts communication events
 * to connect host-side GPU runtime calls with their corresponding GPU kernel
 * and memcopy executions.
 *
 * @param thread_id        Buffer thread id
 * @param time             Timestamp of the event
 * @param event            Kind of event (e.g. GPU(KERNEL_GPU_VAL))
 * @param value            Entry/exit (EVT_BEGIN / EVT_END)
 * @param tag              Communication tag
 * @param size             Size in bytes for memory operations
 * @param blockspergrid    Grid configuration (GPU kernels)
 * @param threadsperblock  Block configuration (GPU kernels)
 * ══════════════════════════════════════════════════════════════════════════ */
static void traceGPUEvents(int thread_id, UINT64 time, unsigned event, unsigned long long value, unsigned tag, size_t size, unsigned blockspergrid, unsigned threadsperblock)
{
	TRACE_GPU_EVENT(thread_id, time, GPUEV(CALLGPU_EV), event, value, size);

	if (event == GPUEV(KERNEL_GPU_VAL))
	{
		TRACE_GPU_KERNEL_EVENT(thread_id, time, GPUEV(_KERNEL_EXEC_EV), value, blockspergrid, threadsperblock);
		if (value != EVT_END)
			THREAD_TRACE_USER_COMMUNICATION_EVENT(thread_id, time, USER_RECV_EV, TASKID, 0, tag, tag);
	}
	else
	{
		if (tag > 0)
			THREAD_TRACE_USER_COMMUNICATION_EVENT(thread_id, time, (value == EVT_END) ? USER_RECV_EV : USER_SEND_EV, TASKID, size, tag, tag);
	}
}

static void FlushGPUEvents(int device_id, int stream_idx)
{
	int thread_id = deviceArray[device_id].streams[stream_idx].thread_id;
	UINT64 utmp, last_time = 0;
	float ftmp;
	gpu_event_t *gpu_event;
	GPU_EVENT_T *reference_event;
	UINT64 reference_time = 0;
	struct RegisteredStream_t *registered_stream = &deviceArray[device_id].streams[stream_idx];

	gpu_event_t *last_gpu_event = gpuEventList_peek_tail(&registered_stream->gpu_event_list);

	if (last_gpu_event == NULL)
		return;

	GPU_RUNTIME_CHECK(GPU_EVENT_SYNCHRONIZE(last_gpu_event->ts_event));

	reference_event = &registered_stream->device_reference_event->ts_event;
	reference_time = registered_stream->host_reference_time;

	gpu_event = gpuEventList_pop(&registered_stream->gpu_event_list);
	while (gpu_event != NULL)
	{

		if (gpu_event->timetype == EXTRAE_GPU_NEW_TIME)
		{
			GPU_RUNTIME_CHECK(GPU_EVENT_ELAPSED_TIME(&ftmp, *reference_event, gpu_event->ts_event));
			ftmp *= 1000000.0f;
			utmp = reference_time + (UINT64)ftmp;

			reference_event = &gpu_event->ts_event;
			reference_time = utmp;
		}
		else
		{
			utmp = last_time;
		}

		traceGPUEvents(thread_id, utmp, gpu_event->event, gpu_event->value, gpu_event->tag, gpu_event->memSize, gpu_event->blocksPerGrid, gpu_event->threadsPerBlock);
		last_time = utmp;

		if (gpuEventList_isempty(&registered_stream->gpu_event_list))
		{
			gpuEventList_add(&deviceArray[device_id].available_events, registered_stream->device_reference_event);
			registered_stream->device_reference_event = gpu_event;
			registered_stream->host_reference_time = last_time;
			gpu_event = NULL;
		}
		else
		{
			gpuEventList_add(&deviceArray[device_id].available_events, gpu_event);
			gpu_event = gpuEventList_pop(&registered_stream->gpu_event_list);
		}
	}
}

void FlushStreams(int device_id, int stream_idx)
{
	int d, s;
	if (deviceArray == NULL)
		return;

	for (d = (device_id == XTR_FLUSH_ALL_DEVICES ? 0 : device_id); d < (device_id == XTR_FLUSH_ALL_DEVICES ? deviceCount : device_id + 1); ++d)
	{
		if (!deviceArray[d].initialized)
			continue;
		for (s = (stream_idx == XTR_FLUSH_ALL_STREAMS ? 0 : stream_idx); s < (stream_idx == XTR_FLUSH_ALL_STREAMS ? deviceArray[d].num_streams : stream_idx + 1); ++s)
		{
			FlushGPUEvents(d, s);
		}
	}
}

void UnregisterStream(int device_id, GPU_STREAM_T stream)
{
	int stream_idx = SearchStream(device_id, stream);
	if (stream_idx == -1)
		return;

#ifdef DEBUG
	fprintf(stderr, "UnregisterStream(device_id=%d, stream=%p) unassigned from stream_idx=%d/%d\n",
			device_id, stream, stream_idx, deviceArray[device_id].num_streams);
#endif

	FlushStreams(device_id, stream_idx);

	int num_streams = deviceArray[device_id].num_streams - 1;
	struct RegisteredStream_t *rs_tmp = (struct RegisteredStream_t *)xmalloc(num_streams * sizeof(struct RegisteredStream_t));

	memmove(rs_tmp, deviceArray[device_id].streams, stream_idx * sizeof(struct RegisteredStream_t));
	memmove(rs_tmp + stream_idx, deviceArray[device_id].streams + stream_idx + 1, (deviceArray[device_id].num_streams - stream_idx - 1) * sizeof(struct RegisteredStream_t));

	deviceArray[device_id].num_streams = num_streams;
	xfree(deviceArray[device_id].streams);
	deviceArray[device_id].streams = rs_tmp;
}

void AddEventToStream(Extrae_GPU_Time_Type timetype, int device_id, int stream_idx, unsigned event, unsigned long long value, unsigned tag, size_t size,	unsigned int blockspergrid, unsigned int threadsperblock)
{
	struct RegisteredStream_t *registered_stream = &deviceArray[device_id].streams[stream_idx];

	gpu_event_t *gpu_event = gpuEventList_pop(&deviceArray[device_id].available_events);

	gpu_event->event = event;
	gpu_event->value = value;
	gpu_event->tag = tag;
	gpu_event->memSize = size;
	gpu_event->blocksPerGrid = blockspergrid;
	gpu_event->threadsPerBlock = threadsperblock;
	gpu_event->timetype = timetype;

	GPU_RUNTIME_CHECK(GPU_EVENT_RECORD(gpu_event->ts_event, registered_stream->stream));

	gpuEventList_add(&registered_stream->gpu_event_list, gpu_event);
}

/* ══════════════════════════════════════════════════════════════════════════
 * Extrae_gpuFinalize
 *
 * Performing the flush of stream events during Extrae finalization (destructor
 * call) triggers GPU calls after the runtime library has been unloaded, causing
 * crashes.  To avoid this, finalization is always registered with atexit() and
 * gpuInitialized is cleared to 0 to prevent re-entry after main() returns.
 * ══════════════════════════════════════════════════════════════════════════ */
void Extrae_gpuFinalize(void)
{
	Backend_Enter_Instrumentation();

	if (EXTRAE_INITIALIZED() && gpuInitialized == 1)
	{

		for (int i = 0; i < GPU_MAP_NUM_BUCKETS; ++i)
		{
			KernelMapEntry *n = kernel_map_buckets[i];
			while (n)
			{
				KernelMapEntry *nxt = n->next;
				int lineno = 0;
				Extrae_AddFunctionDefinitionEntryToLocalSYM('Y',(void*)(uintptr_t)n->function_handle, n->kernel_name, "??", lineno);
				n = nxt;
			}
		}

		FlushStreams(XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);

		for (int i = 0; i < deviceCount; ++i)
		{
			if (deviceArray[i].initialized == TRUE)
			{
				gpuEventList_free(&deviceArray[i].available_events);
				DeinitializeDevice(i);
			}
		}

		xfree(deviceArray);
		deviceArray = NULL;
		deviceCount = 0;
		gpuInitialized = 0;
	}

	Backend_Leave_Instrumentation();
}