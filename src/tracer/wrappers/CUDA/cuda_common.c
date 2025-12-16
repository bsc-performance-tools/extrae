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
#include "debug.h"

#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#include <cuda_runtime_api.h> 
#ifdef HAVE_PTHREAD_H
# include <pthread.h>
#endif

#include "cuda_common.h"
#include "taskid.h"
#include "threadinfo.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "trace_macros_gpu.h"
#include "cuda_probe.h"
#include "xalloc.h"
#include "extrae_user_events.h"
#include "gpu_event_info.h"
#include "trace_mode.h"

#define SearchStream(device_id, stream) SearchAndRegisterStream(device_id, stream, FALSE)
#define RegisterStream(device_id, stream) SearchAndRegisterStream(device_id, stream, TRUE)

#if defined(__APPLE__)
# define HOST_NAME_MAX 512
#endif

/**
 * Variable to control the initialization and allow the flushing of
 * events at the exit point.
 */
int cudaInitialized = 0;

/* Structures that will hold the parameters needed for the exit parts of the
 * instrumentation code. This way we can support dyninst/ld-preload/cupti
 * instrumentation with a single file
 */
typedef struct
{
	int stream_idx;
	cudaStream_t stream;
	enum cudaMemcpyKind memcpyKind;
	size_t memcpySize;
	unsigned tag;
} CUDA_thread_args_t;

static __thread CUDA_thread_args_t CUDA_thread_args;

unsigned cuda_events_block_size = DEFAULT_CUDA_EVENTS_BLOCK_SIZE;

static unsigned lastTag = 0xC0DA; /* Fixed tag */

static pthread_mutex_t lastTagMutex = PTHREAD_MUTEX_INITIALIZER;

static unsigned GetCUDACommTag()
{
	unsigned new_tag = 0;

	pthread_mutex_lock(&lastTagMutex);
	new_tag = lastTag++;
	pthread_mutex_unlock(&lastTagMutex);

	return new_tag;
}

static struct DeviceInfo_t *deviceArray = NULL;
static int deviceCount = 0;

static void SynchronizeStream(int device_id, int stream_idx)
{
	if (device_id >= deviceCount)
	{
		fprintf(stderr, PACKAGE_NAME": Error! Invalid CUDA device id in SynchronizeStream\n");
		exit(-1);
	}

#ifdef DEBUG
	fprintf (stderr, "SynchronizeStream (device_id=%d, stream_idx=%d, stream=%p)\n", device_id, stream_idx, deviceArray[device_id].streams[stream_idx].stream);
#endif

	if (deviceArray[device_id].streams[stream_idx].device_reference_event == NULL){
		deviceArray[device_id].streams[stream_idx].device_reference_event = gpuEventList_pop(&deviceArray[device_id].available_events);
	}

	CUDA_RUNTIME_CHECK(cudaEventRecord(deviceArray[device_id].streams[stream_idx].device_reference_event->ts_event, deviceArray[device_id].streams[stream_idx].stream));
	CUDA_RUNTIME_CHECK(cudaEventSynchronize(deviceArray[device_id].streams[stream_idx].device_reference_event->ts_event));
	deviceArray[device_id].streams[stream_idx].host_reference_time = TIME;
}

static void DeinitializeDevice(int device_id)
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

static int SearchAndRegisterStream(int device_id, cudaStream_t stream, int register_stream)
{
	int stream_idx = 0;
    unsigned long long unique_stream_id = 0;
	int i = 0;

	/* If deviceArray is not initialized, create it first */
	if (deviceArray == NULL)
	{
		CUDA_RUNTIME_CHECK(cudaGetDeviceCount(&deviceCount));

		deviceArray = (struct DeviceInfo_t*) xmalloc(sizeof(struct DeviceInfo_t)*deviceCount);

		for (i = 0; i < deviceCount; i++)
		{
            deviceArray[i].initialized = FALSE;
            deviceArray[i].streams = NULL;
            deviceArray[i].num_streams = 0;
        }
        /* Register finalization once, at process exit */
        atexit(Extrae_CUDA_finalize);
        cudaInitialized = 1;
    }

	/* Initialize per-device structures on first use */
	if (!deviceArray[device_id].initialized){
        gpuEventList_init(&deviceArray[device_id].available_events, TRUE, XTR_CUDA_EVENTS_BLOCK_SIZE);
		deviceArray[device_id].initialized = TRUE;
	}

	/* Obtain a unique identifier for the CUDA stream */
	CUDA_RUNTIME_CHECK(cudaStreamGetId(stream, &unique_stream_id));
	struct RegisteredStream_t *StreamArray = deviceArray[device_id].streams;

	/* Search for an already registered stream with the same identifier */
	for (stream_idx = 0; stream_idx < deviceArray[device_id].num_streams; stream_idx++){
		if (StreamArray[stream_idx].stream_id == unique_stream_id){
			return stream_idx;
		}
	}

	/* If the stream is not found, register it only if requested */
    if (register_stream) 
	{
		Probe_Cuda_StreamRegister_Entry();

		/* Realloc and refresh StreamArray */
		StreamArray = deviceArray[device_id].streams = (struct RegisteredStream_t *) xrealloc(
			StreamArray, (stream_idx + 1) * sizeof(struct RegisteredStream_t));

		stream_idx = deviceArray[device_id].num_streams;
		deviceArray[device_id].num_streams++;

		/*
		* XXX Should this be Backend_getMaximumOfThreads()? If we
		* previously increased the number of threads in another runtime,
		* and then decreased them, we will end up with a line with mixed
		* semantics (thread&stream).
		*/
		Backend_ChangeNumberOfThreads(Backend_getNumberOfThreads() + 1);

		StreamArray[stream_idx].host_reference_time = 0;
		StreamArray[stream_idx].device_reference_event = NULL;
		StreamArray[stream_idx].thread_id = Backend_getNumberOfThreads()-1;
		StreamArray[stream_idx].stream = stream;
		StreamArray[stream_idx].stream_id = unique_stream_id;
		gpuEventList_init(&StreamArray[stream_idx].gpu_event_list, FALSE, XTR_CUDA_EVENTS_BLOCK_SIZE);

#ifdef DEBUG
		fprintf(stderr, "Extrae_CUDA_RegisterStream(device_id=%d, stream=%p assigned to stream_idx => %d\n", device_id, stream, stream_idx);
#endif
		/* Assign a descriptive name to the thread associated with this stream */
		{
			char _threadname[THREAD_INFO_NAME_LEN];
			char _hostname[HOST_NAME_MAX];

			if (gethostname(_hostname, HOST_NAME_MAX) == 0)
				sprintf(_threadname, "CUDA-D%d.S%d-%s", device_id+1, stream_idx+1, _hostname);
			else
				sprintf(_threadname, "CUDA-D%d.S%d-%s", device_id+1, stream_idx+1, "unknown-host");
			Extrae_set_thread_name(StreamArray[stream_idx].thread_id, _threadname);
		}

		/* Create an event record and process it through the stream! */
		/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
		StreamArray[stream_idx].device_reference_event = NULL;
		SynchronizeStream(device_id, stream_idx);

		/*
		* Necessary to change the base state of CUDA streams from NOT_TRACING
		* to IDLE. We manually emit a TRACING_MODE_DETAIL event because CUDA
		* doesn't have burst mode yet. Will need a revision when supported.
		*/
		THREAD_TRACE_MISCEVENT(StreamArray[stream_idx].thread_id, StreamArray[stream_idx].host_reference_time, 
			TRACING_MODE_EV, TRACE_MODE_DETAIL, 0);

		Probe_Cuda_StreamRegister_Exit();
		return stream_idx;
	}
	/* streams not found and registration was not requested */
	return -1;
}

static void UnregisterStream(int device_id, cudaStream_t stream)
{
	int stream_idx = SearchStream(device_id, stream);

	/* If the stream is not registered, there is nothing to remove */
	if (stream_idx == -1) return;

#ifdef DEBUG
	fprintf(stderr, "UnregisterStream(device_id=%d, stream=%p unassigned from stream_idx => %d/%d\n", device_id, stream, stream_idx, deviceArray[device_id].num_streams);
#endif

	FlushStreams(device_id, stream_idx);

	int num_streams = deviceArray[device_id].num_streams - 1;

	struct RegisteredStream_t *rs_tmp = (struct RegisteredStream_t*) xmalloc(num_streams*sizeof(struct RegisteredStream_t));

	memmove(rs_tmp, deviceArray[device_id].streams, stream_idx * sizeof(struct RegisteredStream_t));
	memmove(rs_tmp+stream_idx, deviceArray[device_id].streams + stream_idx + 1, (deviceArray[device_id].num_streams - stream_idx - 1)*sizeof(struct RegisteredStream_t));

	deviceArray[device_id].num_streams = num_streams;

	xfree(deviceArray[device_id].streams);
	deviceArray[device_id].streams = rs_tmp;
}

static void AddEventToStream(Extrae_CUDA_Time_Type timetype,
	int device_id, int stream_idx, unsigned event, unsigned long long value,
	unsigned tag, size_t size, unsigned int blockspergrid, unsigned int threadsperblock)
{
	struct RegisteredStream_t *registered_stream = &deviceArray[device_id].streams[stream_idx];

	gpu_event_t* gpu_event = gpuEventList_pop(&deviceArray[device_id].available_events);

	gpu_event->event = event;
	gpu_event->value = value;
	gpu_event->tag = tag;
	gpu_event->memSize = size;
	gpu_event->blocksPerGrid = blockspergrid;
	gpu_event->threadsPerBlock = threadsperblock;
	gpu_event->timetype = timetype;

	CUDA_RUNTIME_CHECK(cudaEventRecord(gpu_event->ts_event, registered_stream->stream));

	gpuEventList_add(&registered_stream->gpu_event_list, gpu_event);
}

/**
 * This routine emits GPU events into the tracing buffer and
 * inserts communication events to connect host-side 
 * CUDA runtime calls with their corresponding GPU kernel and memcopies executions.
 *
 * @param thread_id          buffer thread id
 * @param time              Timestamp of the event
 * @param event             kind of event (CUDAKERNEL_GPU_VAL, CUDAMEMCPY_GPU_VAL...)
 * @param value             Entry/exit (EVT_BEGIN/EVT_END)
 * @param tag               Communication tag
 * @param size              Size in bytes for memory operations
 * @param blockspergrid     Grid configuration (CUDA kernels)
 * @param threadsperblock   Block configuration (CUDA kernels)
 */
static void TraceGPUEvents(int thread_id, UINT64 time, unsigned event, unsigned long long value, unsigned tag, size_t size, unsigned blockspergrid, unsigned threadsperblock)
{
	TRACE_GPU_EVENT(thread_id, time, CUDACALLGPU_EV, event, value, size);

	if (event == CUDAKERNEL_GPU_VAL)
	{
		TRACE_GPU_KERNEL_EVENT(thread_id, time, CUDA_KERNEL_EXEC_EV, value, blockspergrid, threadsperblock);
		if(value != EVT_END) 
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(thread_id, time,USER_RECV_EV,TASKID,0, tag, tag);
		}
	}
	else
	{
		if ((event == CUDAMEMCPY_GPU_VAL || event == CUDAMEMCPYASYNC_GPU_VAL) && (tag > 0))
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(thread_id, time, (value==EVT_END)?USER_RECV_EV:USER_SEND_EV, TASKID, size, tag, tag);
		}
	}
}

static void FlushGPUEvents(int device_id, int stream_idx)
{
	int thread_id = deviceArray[device_id].streams[stream_idx].thread_id;
	UINT64 utmp, last_time = 0;
	float ftmp;
	gpu_event_t *gpu_event;
	cudaEvent_t *reference_event;
	UINT64 reference_time = 0;
	struct RegisteredStream_t *registered_stream = &deviceArray[device_id].streams[stream_idx];

	gpu_event_t *last_gpu_event = gpuEventList_peek_tail(&registered_stream->gpu_event_list);

	if(last_gpu_event != NULL)
	{
		CUDA_RUNTIME_CHECK(cudaEventSynchronize(last_gpu_event->ts_event));
		reference_event = &registered_stream->device_reference_event->ts_event;
		reference_time = registered_stream->host_reference_time;

		/* Translate time from GPU to CPU using .device_reference_time and .host_reference_time
				from the RegisteredStream_t structure */

		gpu_event = gpuEventList_pop(&registered_stream->gpu_event_list);
		while(gpu_event != NULL)
		{
			if (gpu_event->timetype == EXTRAE_CUDA_NEW_TIME)
			{
				/* Computes the elapsed time between two events (in ms with a
				* resolution of around 0.5 microseconds) -- according to
				* https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
				*/
				CUDA_RUNTIME_CHECK(cudaEventElapsedTime(&ftmp, *reference_event, gpu_event->ts_event));
				ftmp *= 1000000;
				/* Time correction between CPU & GPU */
				utmp = reference_time + (UINT64) (ftmp);

				reference_event = &gpu_event->ts_event;
				reference_time = utmp;
			}
			else
			{
				utmp = last_time;
			}

			traceGPUEvents(thread_id, utmp, gpu_event->event, gpu_event->value, gpu_event->tag, gpu_event->memSize, gpu_event->blocksPerGrid, gpu_event->threadsPerBlock);

			last_time = utmp;

			if(gpuEventList_isempty(&registered_stream->gpu_event_list))
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
}

void FlushStreams(int device_id, int stream_idx)
{
	int d = 0, s = 0;

	if (deviceArray != NULL)
	{
		for (d = (device_id == XTR_FLUSH_ALL_DEVICES ? 0 : device_id);
			 d < (device_id == XTR_FLUSH_ALL_DEVICES ? deviceCount : device_id+1);
			++d)
		{
			if (deviceArray[d].initialized)
			{
				for (s = (stream_idx == XTR_FLUSH_ALL_STREAMS ? 0 : stream_idx);
					 s < (stream_idx == XTR_FLUSH_ALL_STREAMS ? deviceArray[d].num_streams : stream_idx+1);
					 ++s)
				{
					FlushGPUEvents(d, s);
				}
			}
		}
	}
}

/****************************************************************************/
/* CUDA INSTRUMENTATION                                                     */
/****************************************************************************/

void Extrae_cudaConfigureCall_Enter(void)
{
	Probe_Cuda_ConfigureCall_Entry();
}

void Extrae_cudaConfigureCall_Exit(void)
{
	Probe_Cuda_ConfigureCall_Exit();
}

void Extrae_cudaLaunch_Enter(const char *f, unsigned int blocksPerGrid, unsigned int threadsPerBlock,
                             size_t sharedMemBytes, cudaStream_t stream, CUcontext ctx)
{
	int device_id = -1; 
	int stream_idx = -1;
	unsigned tag = GetCUDACommTag();
	CUDA_thread_args.tag = tag;

	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	stream_idx = RegisterStream(device_id, stream);
	CUDA_thread_args.stream_idx = stream_idx;

	Probe_Cuda_Launch_Entry((UINT64) f);

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
	  CUDAKERNEL_GPU_VAL, (UINT64)f, tag, sharedMemBytes, blocksPerGrid, threadsPerBlock);
}

void Extrae_cudaLaunch_Exit(CUcontext ctx)
{
	int device_id = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);
	unsigned tag = CUDA_thread_args.tag;

	AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, CUDA_thread_args.stream_idx,
		CUDAKERNEL_GPU_VAL, EVT_END, tag, 0, 0, 0);

	Probe_Cuda_Launch_Exit();
}

void Extrae_cudaMalloc_Enter(unsigned int event, void **devPtr, size_t size)
{
	Probe_Cuda_Malloc_Entry(event, (UINT64)devPtr, size);
}

void Extrae_cudaMalloc_Exit(unsigned int event)
{
	Probe_Cuda_Malloc_Exit(event);
}

void Extrae_cudaFree_Enter(unsigned int event, void *devPtr)
{
	Probe_Cuda_Free_Entry(event, (UINT64)devPtr);
}

void Extrae_cudaFree_Exit(unsigned int event)
{
	Probe_Cuda_Free_Exit(event);
}

void Extrae_cudaHostAlloc_Enter(void **pHost, size_t size)
{
	Probe_Cuda_HostAlloc_Entry((UINT64)pHost, size);
}

void Extrae_cudaHostAlloc_Exit()
{
	Probe_Cuda_HostAlloc_Exit();
}

void Extrae_cudaDeviceSynchronize_Enter(CUcontext ctx)
{
	int device_id = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	Probe_Cuda_ThreadBarrier_Entry();
	FlushStreams(device_id, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_cudaDeviceSynchronize_Exit(void)
{
	Probe_Cuda_ThreadBarrier_Exit ();
}

void Extrae_cudaThreadSynchronize_Enter(CUcontext ctx)
{
	int device_id = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	Probe_Cuda_ThreadBarrier_Entry();
	FlushStreams(device_id, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_cudaThreadSynchronize_Exit(void)
{
	Probe_Cuda_ThreadBarrier_Exit();
}

void Extrae_cudaStreamCreate_Enter(void)
{
	Probe_Cuda_StreamCreate_Entry();
}

void Extrae_cudaStreamCreate_Exit(void)
{
	Probe_Cuda_StreamCreate_Exit();
}

void Extrae_cudaStreamDestroy_Enter(cudaStream_t stream, CUcontext ctx)
{
	int device_id = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	Probe_Cuda_StreamDestroy_Entry();

	UnregisterStream(device_id, stream);
}

void Extrae_cudaStreamDestroy_Exit(void)
{
	Probe_Cuda_StreamDestroy_Exit();
}

void Extrae_cudaStreamSynchronize_Enter(cudaStream_t stream, CUcontext ctx)
{
	int device_id = -1;
	int stream_idx = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);
	stream_idx = RegisterStream(device_id, stream);

	Probe_Cuda_StreamBarrier_Entry(deviceArray[device_id].streams[stream_idx].thread_id);
	FlushStreams(device_id, stream_idx);
}

void Extrae_cudaStreamSynchronize_Exit(void)
{
	Probe_Cuda_StreamBarrier_Exit();
}

void _Extrae_cudaMemcpy_Enter(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, 
	                           CUcontext ctx, void(*entry_probe)(size_t), unsigned long long gpu_value)
{
	UNREFERENCED_PARAMETER(dst);
	UNREFERENCED_PARAMETER(src);

	int device_id = -1;
	int stream_idx = -1;
	unsigned tag = GetCUDACommTag();

	CUDA_thread_args.memcpyKind = kind;
	CUDA_thread_args.memcpySize = count;
	CUDA_thread_args.tag = tag;

	CUDA_GET_DEVICE_SAFE(ctx, device_id);
	stream_idx = RegisterStream(device_id, cudaStreamDefault);
	CUDA_thread_args.stream_idx = stream_idx;

	entry_probe(count);

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (kind == cudaMemcpyHostToDevice)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, count, tag, tag);
	}

	/* If the memcpy is started at host, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Exit
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication starts at this point.
	 */
	if (kind == cudaMemcpyDeviceToHost)
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
		  gpu_value, EVT_BEGIN, tag, count, 0, 0);
	}
	else
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
		  gpu_value, EVT_BEGIN, 0, count, 0, 0);
	}
}

void _Extrae_cudaMemcpy_Exit(CUcontext ctx, void (*exit_probe)(void), unsigned long long gpu_value)
{
	int device_id = -1;
	int stream_idx = CUDA_thread_args.stream_idx;
	enum cudaMemcpyKind kind = CUDA_thread_args.memcpyKind;
	size_t size = CUDA_thread_args.memcpySize;
	unsigned tag = CUDA_thread_args.tag;

	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	/* THIS IS SYMMETRIC TO Extrae_cudaMemcpy_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Enter
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication arrives at this point.
	 */
	if (kind == cudaMemcpyHostToDevice)
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
			gpu_value, EVT_END, tag, size, 0, 0);
	}
	else
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
			gpu_value, EVT_END, 0, size, 0, 0);
	}
		
	exit_probe();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (kind == cudaMemcpyDeviceToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID, size, tag, tag);
	}
}

void Extrae_cudaMemcpyAsync_Enter(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream, CUcontext ctx)
{
	UNREFERENCED_PARAMETER(dst);
	UNREFERENCED_PARAMETER(src);

	int device_id = -1;
	int stream_idx = -1;
	unsigned tag = GetCUDACommTag();

	CUDA_thread_args.stream = stream;
	CUDA_thread_args.memcpyKind = kind;
	CUDA_thread_args.memcpySize = count;
	CUDA_thread_args.tag = tag;

	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	Probe_Cuda_MemcpyAsync_Entry(count);

	stream_idx = RegisterStream(device_id, stream);
	CUDA_thread_args.stream_idx = stream_idx;

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (kind == cudaMemcpyHostToDevice)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, count, tag, tag);
	}

	/* If the memcpy is started at host, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Exit
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication starts at this point.
	 */
	if (kind == cudaMemcpyDeviceToHost)
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_BEGIN, tag, count, 0, 0);
	}
	else
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_BEGIN, 0, count, 0, 0);
	}
}

void Extrae_cudaMemcpyAsync_Exit(CUcontext ctx)
{
	int device_id = -1;
	int stream_idx = CUDA_thread_args.stream_idx;
	enum cudaMemcpyKind kind = CUDA_thread_args.memcpyKind;
	size_t size = CUDA_thread_args.memcpySize;
	unsigned tag = CUDA_thread_args.tag;

	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	/* THIS IS SYMMETRIC TO Extrae_cudaMemcpyAsync_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Enter
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication arrives at this point.
	 */
	if (kind == cudaMemcpyHostToDevice)
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_END, tag, size, 0, 0);
	}
	else
	{
		AddEventToStream(EXTRAE_CUDA_NEW_TIME, device_id, stream_idx,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_END, 0, size, 0, 0);
	}

	Probe_Cuda_MemcpyAsync_Exit();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (kind == cudaMemcpyDeviceToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, size, tag, tag);
	}
}

void Extrae_cudaMemset_Enter(void *devPtr, size_t count)
{
	Probe_Cuda_Memset_Entry((UINT64)devPtr, count);
}

void Extrae_cudaMemset_Exit()
{
	Probe_Cuda_Memset_Exit();
}

void Extrae_cudaMemsetAsync_Enter(void *devPtr, size_t count)
{
	Probe_Cuda_MemsetAsync_Entry((UINT64)devPtr, count);
}

void Extrae_cudaMemsetAsync_Exit()
{
	Probe_Cuda_MemsetAsync_Exit();
}

void Extrae_cudaDeviceReset_Enter()
{
	Probe_Cuda_DeviceReset_Enter();
	FlushStreams(XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_cudaDeviceReset_Exit(CUcontext ctx)
{
	int device_id = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	DeinitializeDevice(device_id);
	Probe_Cuda_DeviceReset_Exit();
}

void Extrae_cudaThreadExit_Enter()
{
	Probe_Cuda_ThreadExit_Enter();
}

void Extrae_cudaThreadExit_Exit(CUcontext ctx)
{
	int device_id = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	DeinitializeDevice(device_id);
	Probe_Cuda_ThreadExit_Exit();
}

void Extrae_cudaEventRecord_Enter(cudaEvent_t event, cudaStream_t stream, CUcontext ctx)
{
	int device_id = -1; 
	int stream_idx = -1;
	CUDA_GET_DEVICE_SAFE(ctx, device_id);

	stream_idx = RegisterStream(device_id, stream);
	Probe_Cuda_EventRecord_Entry((UINT64)event, deviceArray[device_id].streams[stream_idx].thread_id);
}

void Extrae_cudaEventRecord_Exit()
{
	Probe_Cuda_EventRecord_Exit();
}

void Extrae_cudaEventSynchronize_Enter(cudaEvent_t event)
{
	Probe_Cuda_EventSynchronize_Entry((UINT64)event);
}

void Extrae_cudaEventSynchronize_Exit()
{
	Probe_Cuda_EventSynchronize_Exit();
}

void Extrae_cudaStreamWaitEvent_Enter()
{
	Probe_Cuda_StreamWaitEvent_Enter();
}

void Extrae_cudaStreamWaitEvent_Exit()
{
	Probe_Cuda_StreamWaitEvent_Exit();
}

/**
 * Performing the flush of streams events during the finalization of Extrae in the destructor call
 * triggers CUDA calls after the CUDA library has been unloaded resulting in crashes.
 * To solve that case we always call Extrae_CUDA_finalize with atexit and mark cudaInitialized to 0
 * to prevent entering this routine after the return point of the main.
 */
void Extrae_CUDA_finalize(void)
{
	Backend_Enter_Instrumentation();
	if (EXTRAE_INITIALIZED() && cudaInitialized == 1)
	{
		FlushStreams(XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);

		for(int i = 0; i < deviceCount ; ++i){
			if(deviceArray[i].initialized == TRUE){
				gpuEventList_free(&deviceArray[i].available_events);
				DeinitializeDevice(i);
			}
		}
		xfree(deviceArray);
		deviceArray = NULL;
		deviceCount = 0;
		cudaInitialized = 0;
	}
	Backend_Leave_Instrumentation();
}
