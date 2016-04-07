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
#include "cuda_probe.h"

#if defined(__APPLE__)
# define HOST_NAME_MAX 512
#endif

/* Structures that will hold the parameters needed for the exit parts of the
   instrumentation code. This way we can support dyninst/ld-preload/cupti
   instrumentation with a single file */

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
} Extrae_cuda_saved_params_t;

static Extrae_cuda_saved_params_t *Extrae_CUDA_saved_params = NULL;

static unsigned __last_tag = 0xC0DA; /* Fixed tag */
static unsigned Extrae_CUDA_tag_generator (void)
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

	err = cudaEventRecord (devices[devid].Stream[streamid].device_reference_time,
		devices[devid].Stream[streamid].stream);
	CHECK_CU_ERROR(err, cudaEventRecord);

	err = cudaEventSynchronize (devices[devid].Stream[streamid].device_reference_time);
	CHECK_CU_ERROR(err, cudaEventSynchronize);

	devices[devid].Stream[streamid].host_reference_time = TIME;
}

static void Extrae_CUDA_deInitialize (int devid)
{
	if (devices != NULL)
	{
		if (devices[devid].initialized)
		{
			free (devices[devid].Stream);
			devices[devid].Stream = NULL;
			devices[devid].initialized = FALSE;
		}
	}
}

static void Extrae_CUDA_Initialize (int devid)
{
	cudaError_t err;
	int i;

	/* If devices table is not initialized, create it first */
	if (devices == NULL)
	{
		err = cudaGetDeviceCount (&CUDAdevices);
		CHECK_CU_ERROR (err, cudaGetDeviceCount);

		devices = (struct CUDAdevices_t*) malloc (sizeof(struct CUDAdevices_t)*CUDAdevices);
		if (devices == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Cannot allocate information for CUDA devices!\n");
			exit (-1);
		}

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

		devices[devid].Stream = (struct RegisteredStreams_t*) malloc (
		  devices[devid].nstreams*sizeof(struct RegisteredStreams_t));
		if (devices[devid].Stream == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Cannot allocate information for CUDA default stream in device %d!\n", devid);
			exit (-1);
		}

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
			Backend_ChangeNumberOfThreads (Backend_getNumberOfThreads() + 1);
			devices[devid].Stream[0].threadid = Backend_getNumberOfThreads()-1;

			/* Set thread name */
			Extrae_set_thread_name (devices[devid].Stream[0].threadid, _threadname);
		}

		/* default device stream */
		devices[devid].Stream[0].stream = (cudaStream_t) 0;
		devices[devid].Stream[0].nevents = 0;

		/* Create an event record and process it through the stream! */
		/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
		err = cudaEventCreateWithFlags (&(devices[devid].Stream[0].device_reference_time), 0);
		CHECK_CU_ERROR (err, cudaEventCreateWithFlags);

		Extrae_CUDA_SynchronizeStream (devid, 0);

		for (i = 0; i < MAX_CUDA_EVENTS; i++)
		{
			/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
			err = cudaEventCreateWithFlags (&(devices[devid].Stream[0].ts_events[i]), 0);
			CHECK_CU_ERROR(err, cudaEventCreateWithFlags);
		}

		devices[devid].initialized = TRUE;
	}
}

static int Extrae_CUDA_SearchStream (int devid, cudaStream_t stream)
{
	int i;

	for (i = 0; i < devices[devid].nstreams; i++)
		if (devices[devid].Stream[i].stream == stream)
			return i;

	return -1;
}

static void Extrae_CUDA_RegisterStream (int devid, cudaStream_t stream)
{
	int i,j, err; 

	i = devices[devid].nstreams;

	devices[devid].Stream = (struct RegisteredStreams_t *) realloc (
	  devices[devid].Stream, (i+1)*sizeof(struct RegisteredStreams_t));

	if (devices[devid].Stream != NULL)
	{
		devices[devid].nstreams++;

		Backend_ChangeNumberOfThreads (Backend_getNumberOfThreads()+1);

		devices[devid].Stream[i].threadid = Backend_getNumberOfThreads()-1;
		devices[devid].Stream[i].stream = stream;
		devices[devid].Stream[i].nevents = 0;

#ifdef DEBUG
		fprintf (stderr, "Extrae_CUDA_RegisterStream (devid=%d, stream=%p assigned to streamid => %d\n", devid, stream, i);
#endif

		/* Set thread name */
		{
			char _threadname[THREAD_INFO_NAME_LEN];
			char _hostname[HOST_NAME_MAX];

			if (gethostname(_hostname, HOST_NAME_MAX) == 0)
				sprintf (_threadname, "CUDA-D%d.S%d-%s", devid+1, i+1, _hostname);
			else
				sprintf (_threadname, "CUDA-D%d.S%d-%s", devid+1, i+1, "unknown-host");
			Extrae_set_thread_name (devices[devid].Stream[i].threadid, _threadname);
		}

		/* Create an event record and process it through the stream! */	
		/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
		err = cudaEventCreateWithFlags (&(devices[devid].Stream[i].device_reference_time), 0);
		CHECK_CU_ERROR(err, cudaEventCreateWithFlags);
		Extrae_CUDA_SynchronizeStream (devid, i);

		for (j = 0; j < MAX_CUDA_EVENTS; j++)
		{
			/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
			err = cudaEventCreateWithFlags (&(devices[devid].Stream[i].ts_events[j]), 0);
			CHECK_CU_ERROR(err, cudaEventCreateWithFlags);
		}
	}	
	else
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot register stream %p on device %d\n", stream, devid);
		exit (-1);
	}
}

static void Extrae_CUDA_AddEventToStream (Extrae_CUDA_Time_Type timetype,
	int devid, int streamid, unsigned event, unsigned long long value,
	unsigned tag, unsigned size)
{
	int evt_index, err;
	struct RegisteredStreams_t *ptr;

	ptr = &devices[devid].Stream[streamid];

	evt_index = ptr->nevents;

	if (evt_index < MAX_CUDA_EVENTS)
	{
#ifdef DEBUG
		fprintf (stderr, "Extrae_CUDA_AddEventToStream (.. devid=%d, streamid=%d, stream=%p .. )\n", devid, streamid, ptr->stream);
#endif
		err = cudaEventRecord (ptr->ts_events[evt_index], ptr->stream);
		CHECK_CU_ERROR(err, cudaEventRecord);

		ptr->events[evt_index] = event;
		ptr->values[evt_index] = value;
		ptr->tag[evt_index] = tag;
		ptr->size[evt_index] = size;
		ptr->timetype[evt_index] = timetype;
		ptr->nevents++;
	}
	else
		fprintf (stderr, PACKAGE_NAME": Warning! Dropping events! Increase MAX_CUDA_EVENTS\n");
}

static void Extrae_CUDA_FlushStream (int devid, int streamid)
{
	int threadid = devices[devid].Stream[streamid].threadid;
	int err;
	unsigned i;
	UINT64 last_time = 0;

	/* Check whether we will fill the buffer soon (or now) */
	if (Buffer_RemainingEvents(TracingBuffer[threadid]) <= 2*devices[devid].Stream[streamid].nevents)
		Buffer_ExecuteFlushCallback (TracingBuffer[threadid]);

	/* Flush events into thread buffer */
	for (i = 0; i < devices[devid].Stream[streamid].nevents; i++)
	{
		UINT64 utmp;
		float ftmp;

		/* Translate time from GPU to CPU using .device_reference_time and .host_reference_time
		   from the RegisteredStreams_t structure */
		err = cudaEventSynchronize (devices[devid].Stream[streamid].ts_events[i]);
		CHECK_CU_ERROR(err, cudaEventSynchronize);

		if (devices[devid].Stream[streamid].timetype[i] == EXTRAE_CUDA_NEW_TIME)
		{
			/* Computes the elapsed time between two events (in ms  with a resolution of
			  around 0.5 microseconds) -- according to
              https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html */
			err = cudaEventElapsedTime (&ftmp,
			  devices[devid].Stream[streamid].device_reference_time,
			  devices[devid].Stream[streamid].ts_events[i]);
			CHECK_CU_ERROR(err, cudaEventElapsedTime);
			ftmp *= 1000000;
			/* Time correction between CPU & GPU */
			utmp = devices[devid].Stream[streamid].host_reference_time + (UINT64) (ftmp);
		}
		else
			utmp = last_time;

		/* Emit events into the tracing buffer.
		    CUDAMEMCPY_GPU_EV & CUDAMEMCPYASYNC_GPU_EV use the size field in THREAD_TRACE_MISCEVENT */
		if (devices[devid].Stream[streamid].events[i] == CUDAMEMCPY_GPU_EV ||
		    devices[devid].Stream[streamid].events[i] == CUDAMEMCPYASYNC_GPU_EV)
		{
			THREAD_TRACE_MISCEVENT (threadid, utmp,
			  devices[devid].Stream[streamid].events[i],
			  devices[devid].Stream[streamid].values[i],
			  devices[devid].Stream[streamid].size[i]);
		}
		else
		{
			THREAD_TRACE_MISCEVENT (threadid, utmp,
			  devices[devid].Stream[streamid].events[i],
			  devices[devid].Stream[streamid].values[i], 0);
		}

		/* Emit communication records for memory transfer, kernel setup and kernel execution */
		if (devices[devid].Stream[streamid].events[i] == CUDAMEMCPY_GPU_EV ||
		    devices[devid].Stream[streamid].events[i] == CUDAMEMCPYASYNC_GPU_EV)
		{
			if (devices[devid].Stream[streamid].tag[i] > 0)
				THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, utmp,
				 (devices[devid].Stream[streamid].values[i]==EVT_END)?USER_RECV_EV:USER_SEND_EV,
				 TASKID,
				 devices[devid].Stream[streamid].size[i],
				 devices[devid].Stream[streamid].tag[i],
				 devices[devid].Stream[streamid].tag[i]);
		}
		else if (devices[devid].Stream[streamid].events[i] == CUDAKERNEL_GPU_EV &&
		         devices[devid].Stream[streamid].values[i] != EVT_END)
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, utmp,
			 USER_RECV_EV,
			 TASKID,
			 0,
			 Extrae_CUDA_tag_generator(),
			 Extrae_CUDA_tag_generator());
		}
		else if (devices[devid].Stream[streamid].events[i] == CUDACONFIGKERNEL_GPU_EV &&
		         devices[devid].Stream[streamid].values[i] != EVT_END)
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, utmp,
			 USER_RECV_EV,
			 TASKID,
			 0,
			 Extrae_CUDA_tag_generator(),
			 Extrae_CUDA_tag_generator());
		}

		last_time = utmp;
	}
	devices[devid].Stream[streamid].nevents = 0;
}

/****************************************************************************/
/* CUDA INSTRUMENTATION                                                     */
/****************************************************************************/

static int _cudaLaunch_stream = 0;

void Extrae_cudaLaunch_Enter (const char *p1)
{
	int devid;
	unsigned tag = Extrae_CUDA_tag_generator();

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_Launch_Entry ((UINT64) p1);

	TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream, CUDAKERNEL_GPU_EV, (UINT64) p1, 0, 0);
}

void Extrae_cudaLaunch_Exit (void)
{
	int devid;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream, CUDAKERNEL_GPU_EV, EVT_END, 0, 0);
	Probe_Cuda_Launch_Exit ();
	Backend_Leave_Instrumentation ();
}

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

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_ConfigureCall_Entry ();

	TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	strid = Extrae_CUDA_SearchStream (devid, p4);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaConfigureCall (p4=%p)\n", p4);
		exit (-1);
	}
	_cudaLaunch_stream = strid;
	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid, CUDACONFIGKERNEL_GPU_EV, EVT_BEGIN, 0, 0);
}

void Extrae_cudaConfigureCall_Exit (void)
{
	int devid;

	cudaGetDevice (&devid);

	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream, CUDACONFIGKERNEL_GPU_EV, EVT_END, 0, 0);
	Probe_Cuda_ConfigureCall_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaDeviceSynchronize_Enter (void)
{
	int devid;
	int i;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_ThreadBarrier_Entry ();

	/* Emit one thread synchronize per stream (begin event) */
	for (i = 0; i < devices[devid].nstreams; i++)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, i,
		  CUDATHREADBARRIER_GPU_EV, EVT_BEGIN, 0, 0);
}

void Extrae_cudaDeviceSynchronize_Exit (void)
{
	int devid;
	int i;

	cudaGetDevice (&devid);

	/* Emit one thread synchronize per stream (end event)*/
	for (i = 0; i < devices[devid].nstreams; i++)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, i,
		  CUDATHREADBARRIER_GPU_EV, EVT_END, 0, 0);

	for (i = 0; i < devices[devid].nstreams; i++)
	{
		Extrae_CUDA_FlushStream (devid, i);
		Extrae_CUDA_SynchronizeStream (devid, i);
	}

	Probe_Cuda_ThreadBarrier_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaThreadSynchronize_Enter (void)
{
	int devid;
	int i;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_ThreadBarrier_Entry ();

	/* Emit one thread synchronize per stream (begin event) */
	for (i = 0; i < devices[devid].nstreams; i++)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, i,
		  CUDATHREADBARRIER_GPU_EV, EVT_BEGIN, 0, 0);
}

void Extrae_cudaThreadSynchronize_Exit (void)
{
	int devid;
	int i;

	cudaGetDevice (&devid);

	/* Emit one thread synchronize per stream (end event)*/
	for (i = 0; i < devices[devid].nstreams; i++)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, i,
		  CUDATHREADBARRIER_GPU_EV, EVT_END, 0, 0);

	for (i = 0; i < devices[devid].nstreams; i++)
	{
		Extrae_CUDA_FlushStream (devid, i);
		Extrae_CUDA_SynchronizeStream (devid, i);
	}

	Probe_Cuda_ThreadBarrier_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_CUDA_flush_all_streams (int synchronize)
{
	int i;
	int devid;

	for (devid = 0; devid < CUDAdevices; devid++)
		if(devices[devid].initialized)
	    		for (i = 0; i < devices[devid].nstreams; i++)
	    		{
	    			Extrae_CUDA_FlushStream (devid, i);
					if (synchronize)
		    			Extrae_CUDA_SynchronizeStream (devid, i);
	    		}
}

void Extrae_cudaStreamCreate_Enter (cudaStream_t *p1)
{
	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].csc.stream = p1;
}

void Extrae_cudaStreamCreate_Exit (void)
{
	int devid;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Extrae_CUDA_RegisterStream (devid,
	  *Extrae_CUDA_saved_params[THREADID].csc.stream);
}

void Extrae_cudaStreamSynchronize_Enter (cudaStream_t p1)
{
	int strid;
	int devid;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].css.stream = p1;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	strid = Extrae_CUDA_SearchStream (devid,
	  Extrae_CUDA_saved_params[THREADID].css.stream);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_StreamBarrier_Entry (devices[devid].Stream[strid].threadid);

	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaStreamSynchronize\n");
		exit (-1);
	}

	/* Emit one thread synchronize per stream (begin event) */
	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid,
	  CUDATHREADBARRIER_GPU_EV, EVT_BEGIN, 0, 0);
}

void Extrae_cudaStreamSynchronize_Exit (void)
{
	int strid;
	int devid; 

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	strid = Extrae_CUDA_SearchStream (devid,
	  Extrae_CUDA_saved_params[THREADID].css.stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaStreamSynchronize\n");
		exit (-1);
	}

	/* Emit one thread synchronize per stream (begin event) */
	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid,
	  CUDATHREADBARRIER_GPU_EV, EVT_END, 0, 0);

	Extrae_CUDA_FlushStream (devid, strid);
	Extrae_CUDA_SynchronizeStream (devid, strid);
	Probe_Cuda_StreamBarrier_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaMemcpy_Enter (void* p1, const void* p2, size_t p3, enum cudaMemcpyKind p4)
{
	int devid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].cm.size = p3;
	Extrae_CUDA_saved_params[THREADID].cm.kind = p4;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_Memcpy_Entry (p3);

	tag = Extrae_CUDA_tag_generator();

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, p3, tag, tag);
	}

	/* If the memcpy is started at host, we use tag = 0 to indicate that we don't want
	   a communication at this point (this will occur at _Exit point).
	   If the memcpy was started at the accelerator, we pass a tag != 0 to indicate that
	   the communication starts at this point. */
	if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_EV, EVT_BEGIN, 0, p3);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_EV, EVT_BEGIN, tag, p3);
}

void Extrae_cudaMemcpy_Exit (void)
{
	int devid;
	int i;
	unsigned tag;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	tag = Extrae_CUDA_tag_generator();

	/* THIS IS SYMMETRIC TO Extrae_cudaMemcpy_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we don't want
	   a communication at this point (this will occur at _Enter point).
	   If the memcpy was started at the accelerator, we pass a tag != 0 to indicate that
	   the communication arrives at this point. */
	if (Extrae_CUDA_saved_params[THREADID].cm.kind == cudaMemcpyHostToDevice ||
	  Extrae_CUDA_saved_params[THREADID].cm.kind == cudaMemcpyDeviceToDevice)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_EV, EVT_END, tag, 0);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0,
		  CUDAMEMCPY_GPU_EV, EVT_END, 0, 0);

	/* This is a safe point because cudaMemcpy is a synchronization point */
	for (i = 0; i < devices[devid].nstreams; i++)
	{
		Extrae_CUDA_FlushStream (devid, i);
		Extrae_CUDA_SynchronizeStream (devid, i);
	}

	Probe_Cuda_Memcpy_Exit ();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (Extrae_CUDA_saved_params[THREADID].cm.kind == cudaMemcpyDeviceToHost ||
	  Extrae_CUDA_saved_params[THREADID].cm.kind == cudaMemcpyHostToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, Extrae_CUDA_saved_params[THREADID].cm.size, tag, tag);
	}

	Backend_Leave_Instrumentation ();
}

void Extrae_cudaMemcpyAsync_Enter (void* p1, const void* p2, size_t p3, enum cudaMemcpyKind p4,cudaStream_t p5)
{
	int devid;
	int strid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	Extrae_CUDA_saved_params[THREADID].cma.size = p3;
	Extrae_CUDA_saved_params[THREADID].cma.kind = p4;
	Extrae_CUDA_saved_params[THREADID].cma.stream = p5;

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_MemcpyAsync_Entry (p3);

	tag = Extrae_CUDA_tag_generator();

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
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

	/* If the memcpy is started at host, we use tag = 0 to indicate that we don't want
	   a communication at this point (this will occur at _Exit point).
	   If the memcpy was started at the accelerator, we pass a tag != 0 to indicate that
	   the communication starts at this point. */
	if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_EV, EVT_BEGIN, 0, p3);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_EV, EVT_BEGIN, tag, p3);
}

void Extrae_cudaMemcpyAsync_Exit (void)
{
	int devid;
	int strid;
	unsigned tag;

	ASSERT(Extrae_CUDA_saved_params!=NULL, "Unallocated Extrae_CUDA_saved_params");

	cudaGetDevice (&devid);
	Extrae_CUDA_Initialize (devid);

	tag = Extrae_CUDA_tag_generator();

	strid = Extrae_CUDA_SearchStream (devid,
	  Extrae_CUDA_saved_params[THREADID].cma.stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_cudaMemcpyAsync_Enter\n");
		exit (-1);
	}

	/* THIS IS SYMMETRIC TO Extrae_cudaMemcpyAsync_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we don't want
	   a communication at this point (this will occur at _Enter point).
	   If the memcpy was started at the accelerator, we pass a tag != 0 to indicate that
	   the communication arrives at this point. */
	if (Extrae_CUDA_saved_params[THREADID].cma.kind == cudaMemcpyHostToDevice ||
	   Extrae_CUDA_saved_params[THREADID].cma.kind == cudaMemcpyDeviceToDevice)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_EV, EVT_END, tag, 0);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid,
		  CUDAMEMCPYASYNC_GPU_EV, EVT_END, 0, 0);

	Probe_Cuda_MemcpyAsync_Exit ();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (Extrae_CUDA_saved_params[THREADID].cma.kind == cudaMemcpyDeviceToHost ||
	  Extrae_CUDA_saved_params[THREADID].cma.kind == cudaMemcpyHostToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, Extrae_CUDA_saved_params[THREADID].cma.size, tag, tag);
	}

	Backend_Leave_Instrumentation ();
}

void Extrae_reallocate_CUDA_info (unsigned nthreads)
{
	Extrae_CUDA_saved_params = (Extrae_cuda_saved_params_t*) realloc (
		Extrae_CUDA_saved_params, sizeof(Extrae_cuda_saved_params_t)*nthreads);

	if (Extrae_CUDA_saved_params == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot reallocate CUDA parameters buffers per thread!\n");
		exit (-1);
	}
}

void Extrae_cudaDeviceReset_Enter (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_Cuda_DeviceReset_Enter();
}

void Extrae_cudaDeviceReset_Exit (void)
{
	int devid;
	cudaGetDevice (&devid);

	Extrae_CUDA_deInitialize (devid);
	Probe_Cuda_DeviceReset_Exit();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaThreadExit_Enter (void)
{
	Backend_Enter_Instrumentation (2);
	Probe_Cuda_ThreadExit_Enter();
}

void Extrae_cudaThreadExit_Exit (void)
{
	int devid;
	cudaGetDevice (&devid);

	Extrae_CUDA_deInitialize (devid);
	Probe_Cuda_ThreadExit_Exit();
	Backend_Leave_Instrumentation ();
}

void Extrae_CUDA_fini (void)
{
	Extrae_CUDA_flush_all_streams (FALSE);
}

