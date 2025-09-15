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
	cudaStream_t *stream;
} cudaStreamCreate_saved_params_t;

typedef struct
{
	cudaStream_t stream;
} cudaStreamSynchronize_saved_params_t;

typedef struct
{
	size_t size;
	enum cudaMemcpyKind kind;
} cudaMemcpy_saved_params_t;

typedef struct
{
	size_t size;
	enum cudaMemcpyKind kind;
	cudaStream_t stream;
} cudaMemcpyAsync_saved_params_t;

typedef union
{
	cudaStreamCreate_saved_params_t csc;
	cudaStreamSynchronize_saved_params_t css;
	cudaMemcpy_saved_params_t cm;
	cudaMemcpyAsync_saved_params_t cma;
} Extrae_cuda_saved_params_union;

typedef struct
{
	Extrae_cuda_saved_params_union punion;
} Extrae_cuda_saved_params_t;

unsigned cuda_events_block_size = DEFAULT_CUDA_EVENTS_BLOCK_SIZE;

static Extrae_cuda_saved_params_t *Extrae_CUDA_saved_params = NULL;

static unsigned __last_tag = 0xC0DA; /* Fixed tag */
static unsigned Extrae_CUDA_tag_generator (void)
{
	__last_tag++;
	return __last_tag;
}

static unsigned Extrae_CUDA_tag_get()
{
	return __last_tag;
}

static struct CUDAdevices_t *devices = NULL;
static int CUDAdevices = 0;

static void Extrae_CUDA_SynchronizeStream (int devid, int streamid)
{
	int err;

	if (devid >= CUDAdevices)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Invalid CUDA device id in CUDASynchronizeStream\n");
		exit (-1);
	}

#ifdef DEBUG
	fprintf (stderr, "Extrae_CUDA_SynchronizeStream (devid=%d, streamid=%d, stream=%p)\n", devid, streamid, devices[devid].Stream[streamid].stream);
#endif

	if(devices[devid].Stream[streamid].device_reference_event == NULL)
	{
		devices[devid].Stream[streamid].device_reference_event = gpuEventList_pop(&devices[devid].availableEvents);
	}

	err = cudaEventRecord (devices[devid].Stream[streamid].device_reference_event->ts_event, devices[devid].Stream[streamid].stream);
	CHECK_CU_ERROR(err, cudaEventRecord);

	err = cudaEventSynchronize (devices[devid].Stream[streamid].device_reference_event->ts_event);
	CHECK_CU_ERROR(err, cudaEventSynchronize);

	devices[devid].Stream[streamid].host_reference_time = TIME;
}

void Extrae_CUDA_deInitialize (int devid)
{
	if (devices != NULL)
	{
		if (devices[devid].initialized)
		{
			xfree (devices[devid].Stream);
			devices[devid].Stream = NULL;
			devices[devid].initialized = FALSE;
		}
	}
}

void Extrae_CUDA_Initialize (int devid)
{
	cudaError_t err;
	int i;

	/* If devices table is not initialized, create it first */
	if (devices == NULL)
	{
		err = cudaGetDeviceCount (&CUDAdevices);
		CHECK_CU_ERROR (err, cudaGetDeviceCount);

		devices = (struct CUDAdevices_t*) xmalloc (sizeof(struct CUDAdevices_t)*CUDAdevices);

		for (i = 0; i < CUDAdevices; i++)
			devices[i].initialized = FALSE;
	}

	/* If the device we're using is not initialized, create its structures */
	if (!devices[devid].initialized)
	{
		char _threadname[THREAD_INFO_NAME_LEN];
		char _hostname[HOST_NAME_MAX];
		unsigned prev_threadid;
		int found;

		devices[devid].nstreams = 1;

		devices[devid].Stream = (struct RegisteredStreams_t*) xmalloc_and_zero (devices[devid].nstreams*sizeof(struct RegisteredStreams_t));

		/* Was the thread created before (i.e. did we executed a cudadevicereset?) */
		if (gethostname(_hostname, HOST_NAME_MAX) == 0)
			sprintf (_threadname, "CUDA-D%d.S%d-%s", devid+1, 1, _hostname);
		else
			sprintf (_threadname, "CUDA-D%d.S%d-%s", devid+1, 1, "unknown-host");
		prev_threadid = Extrae_search_thread_name (_threadname, &found);

		if (found)
		{
			/* If thread name existed, reuse its thread id */
			devices[devid].Stream[0].threadid = prev_threadid;
		}
		else
		{
			/* For timing purposes we change num of threads here instead of doing Backend_getNumberOfThreads() + CUDAdevices*/
			/*
			 * XXX Should this be Backend_getMaximumOfThreads()? If we
			 * previously increased the number of threads in another runtime,
			 * and then decreased them, we will end up with a line with mixed
			 * semantics (thread&stream).
			 */
			Backend_ChangeNumberOfThreads(Backend_getNumberOfThreads() + 1);
			devices[devid].Stream[0].threadid = Backend_getNumberOfThreads()-1;

			/* Set thread name */
			Extrae_set_thread_name(devices[devid].Stream[0].threadid, _threadname);
		}

		/* default device stream */
		devices[devid].Stream[0].stream = (cudaStream_t) 0;
		gpuEventList_init(&devices[devid].Stream[0].gpu_event_list, FALSE, XTR_CUDA_EVENTS_BLOCK_SIZE);
		gpuEventList_init(&devices[devid].availableEvents, TRUE, XTR_CUDA_EVENTS_BLOCK_SIZE);

		/* Create an event record and process it through the stream! */
		/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */

		devices[devid].Stream[0].device_reference_event = NULL;
		Extrae_CUDA_SynchronizeStream (devid, 0);

		/*
		 * Necessary to change the base state of CUDA streams from NOT_TRACING
		 * to IDLE. We manually emit a TRACING_MODE_DETAIL event because CUDA
		 * doesn't have burst mode yet. Will need a revision when supported.
		 */
		THREAD_TRACE_MISCEVENT(devices[devid].Stream[0].threadid, devices[devid].Stream[0].host_reference_time, TRACING_MODE_EV, TRACE_MODE_DETAIL, 0);

		devices[devid].initialized = TRUE;

		/**
		* Last flush of stream events.
		* This initialization occurs in the callback of a CUDA call from the app,
		* therefore this atexit corresponds to the exit point of the main from the app traced.
		* Moving this to CUPTI initialization is not equivalent since that initialization
		* is called from the Extrae constructor, and therefore that atexit corresponds to our library exit point.
		*/
		atexit(Extrae_CUDA_finalize);
		cudaInitialized = 1;
	}
}

static int Extrae_CUDA_SearchStream (int devid, cudaStream_t stream)
{
	int i;

	/* Starting from CUDA 7, CU_STREAM_LEGACY is a new stream handle that uses 
	   an implicit stream with legacy synchronization behavior, just as the 
	   behaviour of stream 0 (default).

		 CU_STREAM_PER_THREAD is assigned to the tid 0. This is allow apps that used it
		 to generate a trace but this could lead to overlapped kernel events if 
		 several threads of the same parallel use CU_STREAM_PER_THREAD concurrently.
	 */
	if (stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD) return 0;

	for (i = 0; i < devices[devid].nstreams; i++)
		if (devices[devid].Stream[i].stream == stream)
			return i;

	return -1;
}

static void Extrae_CUDA_unRegisterStream (int devid, cudaStream_t stream)
{
	int stid = Extrae_CUDA_SearchStream (devid, stream);

#ifdef DEBUG
	fprintf (stderr, "Extrae_CUDA_unRegisterStream (devid=%d, stream=%p unassigned from streamid => %d/%d\n", devid, stream, stid, devices[devid].nstreams);
#endif

	Extrae_CUDA_flush_streams(devid, stid);

	int nstreams = devices[devid].nstreams - 1;

	struct RegisteredStreams_t *rs_tmp = (struct RegisteredStreams_t*) xmalloc (nstreams*sizeof(struct RegisteredStreams_t));

	memmove (rs_tmp, devices[devid].Stream, stid * sizeof(struct RegisteredStreams_t));
	memmove (rs_tmp+stid, devices[devid].Stream + stid + 1, (devices[devid].nstreams - stid - 1)*sizeof(struct RegisteredStreams_t));

	devices[devid].nstreams = nstreams;

	xfree (devices[devid].Stream);
	devices[devid].Stream = rs_tmp;
}

static void Extrae_CUDA_RegisterStream (int devid, cudaStream_t stream)
{
	int i = devices[devid].nstreams;

	devices[devid].Stream = (struct RegisteredStreams_t *) xrealloc (
	  devices[devid].Stream, (i+1)*sizeof(struct RegisteredStreams_t));

	devices[devid].nstreams++;

	/*
	 * XXX Should this be Backend_getMaximumOfThreads()? If we
	 * previously increased the number of threads in another runtime,
	 * and then decreased them, we will end up with a line with mixed
	 * semantics (thread&stream).
	 */
	Backend_ChangeNumberOfThreads(Backend_getNumberOfThreads()+1);

	devices[devid].Stream[i].host_reference_time = 0;
	devices[devid].Stream[i].device_reference_event = NULL;
	devices[devid].Stream[i].threadid = Backend_getNumberOfThreads()-1;
	devices[devid].Stream[i].stream = stream;
	gpuEventList_init(&devices[devid].Stream[i].gpu_event_list, FALSE, XTR_CUDA_EVENTS_BLOCK_SIZE);

#ifdef DEBUG
	fprintf(stderr, "Extrae_CUDA_RegisterStream (devid=%d, stream=%p assigned to streamid => %d\n", devid, stream, i);
#endif

	/* Set thread name */
	{
		char _threadname[THREAD_INFO_NAME_LEN];
		char _hostname[HOST_NAME_MAX];

		if (gethostname(_hostname, HOST_NAME_MAX) == 0)
			sprintf(_threadname, "CUDA-D%d.S%d-%s", devid+1, i+1, _hostname);
		else
			sprintf(_threadname, "CUDA-D%d.S%d-%s", devid+1, i+1, "unknown-host");
		Extrae_set_thread_name(devices[devid].Stream[i].threadid, _threadname);
	}

	/* Create an event record and process it through the stream! */
	/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
	devices[devid].Stream[i].device_reference_event = NULL;
	Extrae_CUDA_SynchronizeStream(devid, i);

	/*
	 * Necessary to change the base state of CUDA streams from NOT_TRACING
	 * to IDLE. We manually emit a TRACING_MODE_DETAIL event because CUDA
	 * doesn't have burst mode yet. Will need a revision when supported.
	 */
	THREAD_TRACE_MISCEVENT(devices[devid].Stream[i].threadid, devices[devid].Stream[i].host_reference_time, TRACING_MODE_EV, TRACE_MODE_DETAIL, 0);
}

static void Extrae_CUDA_AddEventToStream (Extrae_CUDA_Time_Type timetype,
	int devid, int streamid, unsigned event, unsigned long long value,
	unsigned tag, size_t size, unsigned int blockspergrid, unsigned int threadsperblock)
{
	int err;
	struct RegisteredStreams_t *registered_stream = &devices[devid].Stream[streamid];

	gpu_event_t* gpu_event = gpuEventList_pop(&devices[devid].availableEvents);

	gpu_event->event = event;
	gpu_event->value = value;
	gpu_event->tag = tag;
	gpu_event->memSize = size;
	gpu_event->blocksPerGrid = blockspergrid;
	gpu_event->threadsPerBlock = threadsperblock;
	gpu_event->timetype = timetype;

	err = cudaEventRecord (gpu_event->ts_event, registered_stream->stream);
	CHECK_CU_ERROR(err, cudaEventRecord);

	gpuEventList_add(&registered_stream->gpu_event_list, gpu_event);
}

/**
 * This routine emits GPU events into the tracing buffer and
 * inserts communication events to connect host-side 
 * CUDA runtime calls with their corresponding GPU kernel and memcopies executions.
 *
 * @param threadid          buffer thread id
 * @param time              Timestamp of the event
 * @param event             kind of event (CUDAKERNEL_GPU_VAL, CUDAMEMCPY_GPU_VAL...)
 * @param value             Entry/exit (EVT_BEGIN/EVT_END)
 * @param tag               Communication tag
 * @param size              Size in bytes for memory operations
 * @param blockspergrid     Grid configuration (CUDA kernels)
 * @param threadsperblock   Block configuration (CUDA kernels)
 */
static void traceGPUEvents(int threadid, UINT64 time, unsigned event, unsigned long long value, unsigned tag, size_t size, unsigned blockspergrid, unsigned threadsperblock)
{
	TRACE_GPU_EVENT(threadid, time, CUDACALLGPU_EV, event, value, size);

	if (event == CUDAKERNEL_GPU_VAL)
	{
		TRACE_GPU_KERNEL_EVENT(threadid, time, CUDA_KERNEL_EXEC_EV, value, blockspergrid, threadsperblock);
		if(value != EVT_END) 
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, time,USER_RECV_EV,TASKID,0, tag, tag);
		}
	}
	else
	{
		if ((event == CUDAMEMCPY_GPU_VAL || event == CUDAMEMCPYASYNC_GPU_VAL) && (tag > 0))
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, time, (value==EVT_END)?USER_RECV_EV:USER_SEND_EV, TASKID, size, tag, tag);
		}
	}
}

static void flushGPUEvents (int devid, int streamid)
{
	int threadid = devices[devid].Stream[streamid].threadid;
	int err;
	UINT64 utmp, last_time = 0;
	float ftmp;
	gpu_event_t *gpu_event;
	cudaEvent_t *reference_event;
	UINT64 reference_time = 0;
	struct RegisteredStreams_t *registered_stream = &devices[devid].Stream[streamid];

	gpu_event_t *last_gpu_event = gpuEventList_peek_tail(&registered_stream->gpu_event_list);

	if(last_gpu_event != NULL)
	{
		err = cudaEventSynchronize(last_gpu_event->ts_event);
		CHECK_CU_ERROR(err, cudaEventSynchronize);
		reference_event = &registered_stream->device_reference_event->ts_event;
		reference_time = registered_stream->host_reference_time;

		/* Translate time from GPU to CPU using .device_reference_time and .host_reference_time
				from the RegisteredStreams_t structure */

		gpu_event = gpuEventList_pop(&registered_stream->gpu_event_list);
		while(gpu_event != NULL)
		{
			if (gpu_event->timetype == EXTRAE_CUDA_NEW_TIME)
			{
				/* Computes the elapsed time between two events (in ms with a
				* resolution of around 0.5 microseconds) -- according to
				* https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
				*/
				err = cudaEventElapsedTime (&ftmp, *reference_event, gpu_event->ts_event);
				CHECK_CU_ERROR(err, cudaEventElapsedTime);
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

			traceGPUEvents(threadid, utmp, gpu_event->event, gpu_event->value, gpu_event->tag, gpu_event->memSize, gpu_event->blocksPerGrid, gpu_event->threadsPerBlock);

			last_time = utmp;

			if(gpuEventList_isempty(&registered_stream->gpu_event_list))
			{
				gpuEventList_add(&devices[devid].availableEvents, registered_stream->device_reference_event);
				registered_stream->device_reference_event = gpu_event;
				registered_stream->host_reference_time = last_time;
				gpu_event = NULL;
			}
			else
			{
				gpuEventList_add(&devices[devid].availableEvents, gpu_event);
				gpu_event = gpuEventList_pop(&registered_stream->gpu_event_list);
			}
		}
	}
}

/****************************************************************************/
/* CUDA INSTRUMENTATION                                                     */
/****************************************************************************/

__thread int _cudaLaunch_stream = 0;

void Extrae_cudaConfigureCall_Enter (dim3 p1, dim3 p2, size_t p3, cudaStream_t p4)
{
	int strid;
	int devid;
	unsigned tag = Extrae_CUDA_tag_generator();

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);
	UNREFERENCED_PARAMETER(p3);

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Probe_Cuda_ConfigureCall_Entry ();

	TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	strid = Extrae_CUDA_SearchStream (devid, p4);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaConfigureCall (p4=%p)\n", p4);
		exit (-1);
	}
	_cudaLaunch_stream = strid;
	Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, strid,
	  CUDACONFIGKERNEL_GPU_VAL, EVT_BEGIN, tag, tag, 0, 0);
}

void Extrae_cudaConfigureCall_Exit (void)
{
	int devid;

	cudaGetDevice (&devid);

	Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream, 
	  CUDACONFIGKERNEL_GPU_VAL, EVT_END, Extrae_CUDA_tag_get(), 0, 0, 0);

	Probe_Cuda_ConfigureCall_Exit ();
}

void Extrae_cudaLaunch_Enter (const char *f,
    unsigned int blocksPerGrid,
    unsigned int threadsPerBlock,
    size_t sharedMemBytes,
    cudaStream_t stream)
{
	int devid;
	unsigned tag = Extrae_CUDA_tag_generator();

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Probe_Cuda_Launch_Entry ((UINT64) f);

	TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	int strid = Extrae_CUDA_SearchStream (devid, stream);
	_cudaLaunch_stream = strid;

	Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream,
	  CUDAKERNEL_GPU_VAL, (UINT64)f, tag, sharedMemBytes, blocksPerGrid, threadsPerBlock);
}

void Extrae_cudaLaunch_Exit (void)
{
	int devid;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream,
	  CUDAKERNEL_GPU_VAL, EVT_END, Extrae_CUDA_tag_get(), 0, 0, 0);

	Probe_Cuda_Launch_Exit ();
}

void Extrae_cudaMalloc_Enter(unsigned int event, void **devPtr, size_t size)
{
	int devid;

	cudaGetDevice(&devid);
	Extrae_CUDA_Initialize(devid);

	Probe_Cuda_Malloc_Entry(event, (UINT64)devPtr, size);
}

void Extrae_cudaMalloc_Exit(unsigned int event)
{
	int devid;

	cudaGetDevice(&devid);
	Extrae_CUDA_Initialize(devid);

	Probe_Cuda_Malloc_Exit(event);
}

void Extrae_cudaFree_Enter(unsigned int event, void *devPtr)
{
	int devid;

	cudaGetDevice(&devid);
	Extrae_CUDA_Initialize(devid);

	Probe_Cuda_Free_Entry(event, (UINT64)devPtr);
}

void Extrae_cudaFree_Exit(unsigned int event)
{
	int devid;

	cudaGetDevice(&devid);
	Extrae_CUDA_Initialize(devid);

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

void Extrae_cudaDeviceSynchronize_Enter (void)
{
	int devid;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Probe_Cuda_ThreadBarrier_Entry ();
	Extrae_CUDA_flush_streams(devid, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_cudaDeviceSynchronize_Exit (void)
{
	int devid;

	cudaGetDevice (&devid);

	Probe_Cuda_ThreadBarrier_Exit ();
}

void Extrae_cudaThreadSynchronize_Enter (void)
{
	int devid;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Probe_Cuda_ThreadBarrier_Entry ();
	Extrae_CUDA_flush_streams(devid, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_cudaThreadSynchronize_Exit (void)
{
	int devid;

	cudaGetDevice (&devid);

	Probe_Cuda_ThreadBarrier_Exit ();
}

void Extrae_CUDA_flush_streams ( int device_id, int stream_id )
{
	int d = 0, s = 0;

	if ( devices == NULL )
	{
		return;
	}

	for ( d = (device_id == XTR_FLUSH_ALL_DEVICES ? 0: device_id);
			  d < (device_id == XTR_FLUSH_ALL_DEVICES ? CUDAdevices: device_id+1);
			  ++d )
	{
		if ( devices[d].initialized )
		{
			for ( s = (stream_id == XTR_FLUSH_ALL_STREAMS ? 0: stream_id);
						s < (stream_id == XTR_FLUSH_ALL_STREAMS ?  devices[d].nstreams: stream_id+1);
						++s )
			{
				flushGPUEvents (d, s);
			}
		}
	}
}

void Extrae_cudaStreamCreate_Enter (cudaStream_t *p1)
{
	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].punion.csc.stream = p1;

	Probe_Cuda_StreamCreate_Entry ();
}

void Extrae_cudaStreamCreate_Exit (void)
{
	int devid;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Extrae_CUDA_RegisterStream (devid,
	  *Extrae_CUDA_saved_params[THREADID].punion.csc.stream);

	Probe_Cuda_StreamCreate_Exit ();
}

void Extrae_cudaStreamDestroy_Enter (cudaStream_t stream)
{
	int devid;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Probe_Cuda_StreamDestroy_Entry ();

	Extrae_CUDA_saved_params[THREADID].punion.css.stream = stream;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Extrae_CUDA_unRegisterStream (devid, Extrae_CUDA_saved_params[THREADID].punion.css.stream);
}

void Extrae_cudaStreamDestroy_Exit (void)
{
	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Probe_Cuda_StreamDestroy_Exit ();
}

void Extrae_cudaStreamSynchronize_Enter (cudaStream_t p1)
{
	int strid;
	int devid;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].punion.css.stream = p1;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	strid = Extrae_CUDA_SearchStream (devid,
	  Extrae_CUDA_saved_params[THREADID].punion.css.stream);

	Probe_Cuda_StreamBarrier_Entry (devices[devid].Stream[strid].threadid);

	Extrae_CUDA_flush_streams(devid, strid);

	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaStreamSynchronize\n");
		exit (-1);
	}
}

void Extrae_cudaStreamSynchronize_Exit (void)
{
	int strid;
	int devid;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	strid = Extrae_CUDA_SearchStream (devid,
	  Extrae_CUDA_saved_params[THREADID].punion.css.stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaStreamSynchronize\n");
		exit (-1);
	}

	Probe_Cuda_StreamBarrier_Exit ();
}

void Extrae_cudaMemcpy_Enter (void* p1, const void* p2, size_t p3, enum cudaMemcpyKind p4)
{
	int devid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].punion.cm.size = p3;
	Extrae_CUDA_saved_params[THREADID].punion.cm.kind = p4;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Probe_Cuda_Memcpy_Entry (p3);

	tag = Extrae_CUDA_tag_generator();

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (p4 == cudaMemcpyHostToDevice)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, p3, tag, tag);
	}

	/* If the memcpy is started at host, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Exit
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication starts at this point.
	 */
	if (p4 == cudaMemcpyHostToDevice)
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);
	}
	else if (p4 == cudaMemcpyDeviceToHost)
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_VAL, EVT_BEGIN, tag, p3, 0, 0);
	}
}

void Extrae_cudaMemcpy_Exit (void)
{
	int devid;
	unsigned tag;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	tag = Extrae_CUDA_tag_get();

	/* THIS IS SYMMETRIC TO Extrae_cudaMemcpy_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Enter
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication arrives at this point.
	 */
	if (Extrae_CUDA_saved_params[THREADID].punion.cm.kind == cudaMemcpyHostToDevice ||
	  Extrae_CUDA_saved_params[THREADID].punion.cm.kind == cudaMemcpyDeviceToDevice)
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_VAL, EVT_END, tag, Extrae_CUDA_saved_params[THREADID].punion.cm.size, 0, 0);
	}
	else if (Extrae_CUDA_saved_params[THREADID].punion.cm.kind == cudaMemcpyDeviceToHost)
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_VAL, EVT_END, 0, Extrae_CUDA_saved_params[THREADID].punion.cm.size, 0, 0);
	}

	Probe_Cuda_Memcpy_Exit ();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (Extrae_CUDA_saved_params[THREADID].punion.cm.kind == cudaMemcpyDeviceToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, Extrae_CUDA_saved_params[THREADID].punion.cm.size, tag, tag);
	}
}

void Extrae_cudaMemcpyAsync_Enter (void* p1, const void* p2, size_t p3, enum cudaMemcpyKind p4,cudaStream_t p5)
{
	int devid;
	int strid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].punion.cma.size = p3;
	Extrae_CUDA_saved_params[THREADID].punion.cma.kind = p4;
	Extrae_CUDA_saved_params[THREADID].punion.cma.stream = p5;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Probe_Cuda_MemcpyAsync_Entry (p3);

	tag = Extrae_CUDA_tag_generator();

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (p4 == cudaMemcpyHostToDevice)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, p3, tag, tag);
	}

	strid = Extrae_CUDA_SearchStream (devid, p5);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_cudaMemcpyAsync_Enter\n");
		exit (-1);
	}

	/* If the memcpy is started at host, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Exit
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication starts at this point.
	 */
	if (p4 == cudaMemcpyHostToDevice)
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);
	}
	else if (p4 == cudaMemcpyDeviceToHost)
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_BEGIN, tag, p3, 0, 0);
	}
}

void Extrae_cudaMemcpyAsync_Exit (void)
{
	int devid;
	int strid;
	unsigned tag;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	tag = Extrae_CUDA_tag_get();

	strid = Extrae_CUDA_SearchStream (devid,
	  Extrae_CUDA_saved_params[THREADID].punion.cma.stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_cudaMemcpyAsync_Enter\n");
		exit (-1);
	}

	/* THIS IS SYMMETRIC TO Extrae_cudaMemcpyAsync_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Enter
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication arrives at this point.
	 */
	if (Extrae_CUDA_saved_params[THREADID].punion.cma.kind == cudaMemcpyHostToDevice ||
	   Extrae_CUDA_saved_params[THREADID].punion.cma.kind == cudaMemcpyDeviceToDevice)
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_END, tag, Extrae_CUDA_saved_params[THREADID].punion.cma.size, 0, 0);
	}
	else
	{
		Extrae_CUDA_AddEventToStream(EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_VAL, EVT_END, 0, Extrae_CUDA_saved_params[THREADID].punion.cma.size, 0, 0);
	}

	Probe_Cuda_MemcpyAsync_Exit ();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (Extrae_CUDA_saved_params[THREADID].punion.cma.kind == cudaMemcpyDeviceToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, Extrae_CUDA_saved_params[THREADID].punion.cma.size, tag, tag);
	}
}

void Extrae_cudaMemset_Enter(void *devPtr, size_t count)
{
	int devid;

	cudaGetDevice(&devid);
	Extrae_CUDA_Initialize(devid);

	Probe_Cuda_Memset_Entry((UINT64)devPtr, count);
}

void Extrae_cudaMemset_Exit()
{
	int devid;

	cudaGetDevice(&devid);
	Extrae_CUDA_Initialize(devid);

	Probe_Cuda_Memset_Exit();
}

void Extrae_reallocate_CUDA_info (unsigned old_threads, unsigned nthreads)
{
	Extrae_CUDA_saved_params = (Extrae_cuda_saved_params_t*) xrealloc (
		Extrae_CUDA_saved_params, sizeof(Extrae_cuda_saved_params_t)*nthreads);

	memset(&Extrae_CUDA_saved_params[old_threads], 0, sizeof(Extrae_cuda_saved_params_t)*(nthreads-old_threads));

	if (Extrae_CUDA_saved_params == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot reallocate CUDA parameters buffers per thread!\n");
		exit (-1);
	}
}

void Extrae_cudaDeviceReset_Enter (void)
{
	Probe_Cuda_DeviceReset_Enter();
	Extrae_CUDA_flush_streams(XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_cudaDeviceReset_Exit (void)
{
	int devid;
	cudaGetDevice (&devid);

	Extrae_CUDA_deInitialize (devid);
	Probe_Cuda_DeviceReset_Exit();
}

void Extrae_cudaThreadExit_Enter (void)
{
	Probe_Cuda_ThreadExit_Enter();
}

void Extrae_cudaThreadExit_Exit (void)
{
	int devid;
	cudaGetDevice (&devid);

	Extrae_CUDA_deInitialize (devid);
	Probe_Cuda_ThreadExit_Exit();
}

void Extrae_cudaEventRecord_Enter()
{
	Probe_Cuda_EventRecord_Entry();
}

void Extrae_cudaEventRecord_Exit()
{
	Probe_Cuda_EventRecord_Exit();
}

void Extrae_cudaEventSynchronize_Enter()
{
	Probe_Cuda_EventSynchronize_Entry();
}

void Extrae_cudaEventSynchronize_Exit()
{
	Probe_Cuda_EventSynchronize_Exit();
}

/**
 * Performing the flush of streams events during the finalization of Extrae in the destructor call
 * triggers CUDA calls after the CUDA library has been unloaded resulting in crashes.
 * To solve that case we always call Extrae_CUDA_finalize with atexit and mark cudaInitialized to 0
 * to prevent entering this routine after the return point of the main.
 */
void Extrae_CUDA_finalize (void)
{
	Backend_Enter_Instrumentation();
	if (EXTRAE_INITIALIZED() && cudaInitialized == 1 )
	{
		Extrae_CUDA_flush_streams (XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);

		for(int i = 0; i <CUDAdevices ; ++i){
			gpuEventList_free(&devices[i].availableEvents);
			Extrae_CUDA_deInitialize (i);
		}
		xfree(devices);

		devices = NULL;
		CUDAdevices = 0;

		cudaInitialized = 0;
	}
	Backend_Leave_Instrumentation();
}
