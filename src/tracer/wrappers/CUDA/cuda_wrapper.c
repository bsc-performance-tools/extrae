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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_DLFCN_H
# define __USE_GNU
# include <dlfcn.h>
# undef  __USE_GNU
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "wrapper.h"
#include "trace_macros.h"
#include "cuda_probe.h"

/**
 ** CUDA HELPER ROUTINES
 **/

/**
 ** The following lines are convenient hacks to avoid including cuda.h
 **/
typedef int cudaError_t;
typedef enum 
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
} cudaMemcpyKind_t;
struct dim3 { unsigned int x, y, z; };
typedef void * cudaEvent_t;
typedef void * cudaStream_t;
typedef enum CUevent_flags_enum { CU_EVENT_DEFAULT, CU_EVENT_BLOCKING_SYNC, CU_EVENT_DISABLE_TIMING } CUevent_flags;

#define MAX_CUDA_EVENTS 128

struct RegisteredStreams_t
{
	UINT64 host_reference_time;
	cudaEvent_t device_reference_time; /* accessed through cudaEvent_t */
	unsigned threadid; /* In Paraver sense */
	cudaStream_t stream;

	int nevents;
	cudaEvent_t ts_events[MAX_CUDA_EVENTS];
	unsigned events[MAX_CUDA_EVENTS];
	unsigned long long types[MAX_CUDA_EVENTS];
	unsigned tag[MAX_CUDA_EVENTS];
	unsigned size[MAX_CUDA_EVENTS];
};

struct CUDAdevices_t
{
	struct RegisteredStreams_t *Stream;
	int nstreams;
	int initialized;
};

static unsigned __last_tag = 0;
static unsigned CUDA_tag_generator (void)
{
	return ++__last_tag;
}

static struct CUDAdevices_t *devices = NULL;
static int CUDAdevices = 0;

static void CUDASynchronizeStream (int devid, int streamid)
{
	if (devid >= CUDAdevices)
	{
		fprintf (stderr, "Error! Invalid CUDA device id in CUDASynchronizeStream\n");
		exit (-1);
	}

	if (cudaEventRecord (devices[devid].Stream[streamid].device_reference_time,
		devices[devid].Stream[streamid].stream) != 0)
	{
		fprintf (stderr, "Error! Cannot get CUDA reference time\n");
		exit (-1);
	}
	if (cudaEventSynchronize (devices[devid].Stream[streamid].device_reference_time) != 0)
	{
		fprintf (stderr, "Error! Cannot synchronize to CUDA reference time\n");
		exit (-1);
	}
	devices[devid].Stream[streamid].host_reference_time = TIME;
}

static void InitializeCUDA (int devid)
{
	int i;

	/* If devices table is not initialized, create it first */
	if (devices == NULL)
	{

		if (cudaGetDeviceCount (&CUDAdevices) != 0)
		{
			fprintf (stderr, "Error! Querying number of CUDA device count!\n");
			exit (-1);
		}

		devices = (struct CUDAdevices_t*) malloc (sizeof(struct CUDAdevices_t)*CUDAdevices);
		if (devices == NULL)
		{
			fprintf (stderr, "Error! Cannot allocate information for CUDA devices!\n");
			exit (-1);
		}

		for (i = 0; i < CUDAdevices; i++)
			devices[i].initialized = FALSE;
	}

	/* If the device we're using is not initialized, create its structures */
	if (!devices[devid].initialized)
	{
		devices[devid].nstreams = 1;

		devices[devid].Stream = (struct RegisteredStreams_t*) malloc (
		  devices[devid].nstreams*sizeof(struct RegisteredStreams_t));
		if (devices[devid].Stream == NULL)
		{
			fprintf (stderr, "Error! Cannot allocate information for CUDA default stream in device %d!\n", devid);
			exit (-1);
		}

		/* For timing purposes we change num of threads here instead of doing Backend_getNumberOfThreads() + CUDAdevices*/
		Backend_ChangeNumberOfThreads (Backend_getNumberOfThreads() + 1);

		/* default device stream */
		devices[devid].Stream[0].threadid = Backend_getNumberOfThreads()-1;
		devices[devid].Stream[0].stream = NULL;
		devices[devid].Stream[0].nevents = 0;

		/* Create an event record and process it through the stream! */
		/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
		if (cudaEventCreateWithFlags (&(devices[devid].Stream[0].device_reference_time), CU_EVENT_BLOCKING_SYNC) != 0)
		{
			fprintf (stderr, "Error! Cannot create CUDA reference time\n");
			exit (-1);
		}
		CUDASynchronizeStream (devid, 0);

		for (i = 0; i < MAX_CUDA_EVENTS; i++)
			/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
			if (cudaEventCreateWithFlags (&(devices[devid].Stream[0].ts_events[i]), CU_EVENT_BLOCKING_SYNC) != 0)
			{
				fprintf (stderr, "Error! Cannot create CUDA time events\n");
				exit (-1);
			}

		devices[devid].initialized = TRUE;
	}
}

static int SearchCUDAStream (int devid, cudaStream_t stream)
{
	int i;

	for (i = 0; i < devices[devid].nstreams; i++)
		if (devices[devid].Stream[i].stream == stream)
			return i;

	return -1;
}

static void RegisterCUDAStream (cudaStream_t stream)
{
	int i,j,devid; 

	cudaGetDevice (&devid);

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

		/* Create an event record and process it through the stream! */	
		/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
		if (cudaEventCreateWithFlags (&(devices[devid].Stream[i].device_reference_time), CU_EVENT_BLOCKING_SYNC) != 0)
		{
			fprintf (stderr, "Error! Cannot create CUDA reference time\n");
			exit (-1);
		}
		CUDASynchronizeStream (devid, i);

		for (j = 0; j < MAX_CUDA_EVENTS; j++)
			/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
			if (cudaEventCreateWithFlags (&(devices[devid].Stream[i].ts_events[j]), CU_EVENT_BLOCKING_SYNC) != 0)
			{
				fprintf (stderr, "Error! Cannot create CUDA time events\n");
				exit (-1);
			}
	}
	else
	{
		fprintf (stderr, "Error! Cannot register stream %p on device %d\n", stream, devid);
		exit (-1);
	}
}

static void AddEventToStream (int devid, int streamid, unsigned event, unsigned long long type,
	unsigned tag, unsigned size)
{
	int evt_index;
	struct RegisteredStreams_t *ptr;

	cudaGetDevice (&devid);

	ptr = &devices[devid].Stream[streamid];

	evt_index = ptr->nevents;

	if (evt_index < MAX_CUDA_EVENTS)
	{
		if (cudaEventRecord (ptr->ts_events[evt_index], ptr->stream) != 0)
		{
			fprintf (stderr, "Error! Cannot get CUDA reference time\n");
			exit (-1);
		}
		ptr->events[evt_index] = event;
		ptr->types[evt_index] = type;
		ptr->tag[evt_index] = tag;
		ptr->size[evt_index] = size;
		ptr->nevents++;
	}
	else
		fprintf (stderr, "WARNING! Dropping events! Increase MAX_CUDA_EVENTS\n");
}

static void FlushStream (int devid, int streamid)
{
	int threadid = devices[devid].Stream[streamid].threadid;
	int i;

	/* Check whether we will fill the buffer soon (or now) */
	if (Buffer_RemainingEvents(TracingBuffer[threadid]) <= 2*devices[devid].Stream[streamid].nevents)
		Buffer_ExecuteFlushCallback (TracingBuffer[threadid]);

	/* Flush events into thread buffer */
	for (i = 0; i < devices[devid].Stream[streamid].nevents; i++)
	{
		UINT64 utmp;
		float ftmp;

		if (cudaEventSynchronize (devices[devid].Stream[streamid].ts_events[i]) != 0)
		{
			fprintf (stderr, "Error! Cannot synchronize to CUDA reference time\n");
			exit (-1);
		}

		cudaEventElapsedTime (&ftmp,
		  devices[devid].Stream[streamid].device_reference_time,
		  devices[devid].Stream[streamid].ts_events[i]);
		ftmp *= 1000000;
		utmp = devices[devid].Stream[streamid].host_reference_time + (UINT64) (ftmp);
		THREAD_TRACE_MISCEVENT (threadid, utmp,
		  devices[devid].Stream[streamid].events[i],
		  devices[devid].Stream[streamid].types[i], 0);

		if (devices[devid].Stream[streamid].events[i] == CUDAMEMCPY_GPU_EV)
			if (devices[devid].Stream[streamid].tag[i] > 0)
				THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, utmp,
				 (devices[devid].Stream[streamid].types[i]==EVT_END)?USER_RECV_EV:USER_SEND_EV,
				 0,
				 devices[devid].Stream[streamid].size[i],
				 devices[devid].Stream[streamid].tag[i],
				 devices[devid].Stream[streamid].tag[i]);
				/* FIX, unprepared for MPI apps! */
	}
	devices[devid].Stream[streamid].nevents = 0;
}

/**
 ** Regular instrumentation
 **/
static cudaError_t (*real_cudaLaunch)(char*) = NULL;
static cudaError_t (*real_cudaConfigureCall)(struct dim3, struct dim3, size_t, cudaStream_t) = NULL;
static cudaError_t (*real_cudaThreadSynchronize)(void) = NULL;
static cudaError_t (*real_cudaStreamSynchronize)(cudaStream_t) = NULL;
static cudaError_t (*real_cudaMemcpy)(void*,void*,size_t,cudaMemcpyKind_t) = NULL;
static cudaError_t (*real_cudaMemcpyAsync)(void*,void*,size_t,cudaMemcpyKind_t,cudaStream_t) = NULL;
static cudaError_t (*real_cudaStreamCreate)(cudaStream_t*) = NULL;

void cuda_tracing_init(int rank)
{
	real_cudaLaunch = (cudaError_t(*)(char*)) dlsym (RTLD_NEXT, "cudaLaunch");
	if (real_cudaLaunch == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaLaunch in DSOs!!\n");

	real_cudaConfigureCall = (cudaError_t(*)(struct dim3, struct dim3, size_t, cudaStream_t)) dlsym (RTLD_NEXT, "cudaConfigureCall");
	if (real_cudaConfigureCall == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaConfigureCall in DSOs!!\n");

	real_cudaThreadSynchronize = (cudaError_t(*)(void)) dlsym (RTLD_NEXT, "cudaThreadSynchronize");
	if (real_cudaThreadSynchronize == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaThreadSynchronize in DSOs!!\n");

	real_cudaStreamSynchronize = (cudaError_t(*)(cudaStream_t)) dlsym (RTLD_NEXT, "cudaStreamSynchronize");
	if (real_cudaStreamSynchronize == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaStreamSynchronize in DSOs!!\n");

	real_cudaMemcpy = (cudaError_t(*)(void*,void*,size_t,cudaMemcpyKind_t)) dlsym (RTLD_NEXT, "cudaMemcpy");
	if (real_cudaMemcpy == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaMemcpy in DSOs!!\n");

	real_cudaMemcpyAsync = (cudaError_t(*)(void*,void*,size_t,cudaMemcpyKind_t,cudaStream_t)) dlsym (RTLD_NEXT, "cudaMemcpyAsync");
	if (real_cudaMemcpyAsync == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaMemcpyAsync in DSOs!!\n");

	real_cudaStreamCreate = (cudaError_t(*)(cudaStream_t*)) dlsym (RTLD_NEXT, "cudaStreamCreate");
	if (real_cudaStreamCreate == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find cudaStreamCreate in DSOs!!\n");
}

static int _cudaLaunch_device = 0;
static int _cudaLaunch_stream = 0;

cudaError_t cudaLaunch (char *p1)
{
	cudaError_t res;

	if (real_cudaLaunch != NULL && mpitrace_on)
	{
		InitializeCUDA(_cudaLaunch_device);

		Backend_Enter_Instrumentation (2);
		Probe_Cuda_Launch_Entry ((UINT64) p1);

		AddEventToStream (_cudaLaunch_device, _cudaLaunch_stream, CUDAKERNEL_GPU_EV, (UINT64) p1, 0, 0);

		res = real_cudaLaunch (p1);

		AddEventToStream (_cudaLaunch_device, _cudaLaunch_stream, CUDAKERNEL_GPU_EV, EVT_END, 0, 0);

		Probe_Cuda_Launch_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_cudaLaunch != NULL && !mpitrace_on)
	{
		res = real_cudaLaunch (p1);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaLaunch in DSOs! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaConfigureCall (struct dim3 p1, struct dim3 p2, size_t p3, cudaStream_t *p4)
{
	int strid, devid;
	cudaError_t res;

	if (real_cudaConfigureCall != NULL && mpitrace_on)
	{
		cudaGetDevice (&devid);

		InitializeCUDA (devid);

		Backend_Enter_Instrumentation (2);
		Probe_Cuda_ConfigureCall_Entry ();

		strid = SearchCUDAStream (devid, p4);
		if (strid == -1)
		{
			fprintf (stderr, "Error! Cannot determine stream index in cudaConfigureCall (p4=%p)\n", p4);
			exit (-1);
		}

		_cudaLaunch_device = devid;
		_cudaLaunch_stream = strid;

		AddEventToStream (devid, strid, CUDACONFIGKERNEL_GPU_EV, EVT_BEGIN, 0, 0);

		res = real_cudaConfigureCall (p1, p2, p3, p4);

		AddEventToStream (devid, strid, CUDACONFIGKERNEL_GPU_EV, EVT_END, 0, 0);

		Probe_Cuda_ConfigureCall_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_cudaConfigureCall != NULL && !mpitrace_on)
	{
		res = real_cudaConfigureCall (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaConfigureCall in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaStreamCreate (cudaStream_t *p1)
{
	cudaError_t res;
	int devid;

	if (real_cudaStreamCreate != NULL && mpitrace_on)
	{
		cudaGetDevice (&devid);

		InitializeCUDA (devid);

		res = real_cudaStreamCreate (p1);

		RegisterCUDAStream (*p1);
	}
	else if (real_cudaStreamCreate != NULL && !mpitrace_on)
	{
		res = real_cudaStreamCreate (p1);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamCreate in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaMemcpyAsync (void *p1, void *p2 , size_t p3, cudaMemcpyKind_t p4, cudaStream_t p5)
{
	int devid, strid;
	cudaError_t res;
	unsigned tag;

	if (real_cudaMemcpyAsync != NULL && mpitrace_on)
	{
		cudaGetDevice (&devid);

		InitializeCUDA (devid);

		Backend_Enter_Instrumentation (2);
		Probe_Cuda_Memcpy_Entry (p3);

		tag = CUDA_tag_generator();

		if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
		{
			TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
			  0, p3, tag, tag);
		}

		strid = SearchCUDAStream (devid, p5);
		if (strid == -1)
		{
			fprintf (stderr, "Error! Cannot determine stream index in cudaConfigureCall (p5=%p)\n", p5);
			exit (-1);
		}

		if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
			AddEventToStream (devid, strid, CUDAMEMCPY_GPU_EV, p3, 0, 0);
		else
			AddEventToStream (devid, strid, CUDAMEMCPY_GPU_EV, p3, tag, p3);

		res = real_cudaMemcpyAsync (p1, p2, p3, p4, p5);

		if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyDeviceToDevice)
			AddEventToStream (devid, strid, CUDAMEMCPY_GPU_EV, EVT_END, tag, p3);
		else
			AddEventToStream (devid, strid, CUDAMEMCPY_GPU_EV, EVT_END, 0, 0);

		Probe_Cuda_Memcpy_Exit ();

		if (p4 == cudaMemcpyDeviceToHost || p4 == cudaMemcpyHostToHost)
		{
			TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
			  0, p3, tag, tag);
		}

		Backend_Leave_Instrumentation ();
	}
	else if (real_cudaMemcpyAsync != NULL && !mpitrace_on)
	{
		res = real_cudaMemcpyAsync (p1, p2, p3, p4, p5);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaMemcpyAsync in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaMemcpy (void *p1, void *p2 , size_t p3, cudaMemcpyKind_t p4)
{
	int i, devid;
	cudaError_t res;
	unsigned tag;

	if (real_cudaMemcpy != NULL && mpitrace_on)
	{
		cudaGetDevice (&devid);

		InitializeCUDA(devid);

		Backend_Enter_Instrumentation (2);
		Probe_Cuda_Memcpy_Entry (p3);

		tag = CUDA_tag_generator();

		if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
		{
			TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
			  0, p3, tag, tag);
		}

		if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyHostToHost)
			AddEventToStream (devid, 0, CUDAMEMCPY_GPU_EV, p3, 0, 0);
		else
			AddEventToStream (devid, 0, CUDAMEMCPY_GPU_EV, p3, tag, p3);

		res = real_cudaMemcpy (p1, p2, p3, p4);

		if (p4 == cudaMemcpyHostToDevice || p4 == cudaMemcpyDeviceToDevice)
			AddEventToStream (devid, 0, CUDAMEMCPY_GPU_EV, EVT_END, tag, p3);
		else
			AddEventToStream (devid, 0, CUDAMEMCPY_GPU_EV, EVT_END, 0, 0);

		for (i = 0; i < devices[devid].nstreams; i++)
		{
			FlushStream (devid, i);
			CUDASynchronizeStream (devid, i);
		}

		Probe_Cuda_Memcpy_Exit ();

		if (p4 == cudaMemcpyDeviceToHost || p4 == cudaMemcpyHostToHost)
		{
			TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
			  0, p3, tag, tag);
		}

		Backend_Leave_Instrumentation ();
	}
	else if (real_cudaMemcpy != NULL && !mpitrace_on)
	{
		res = real_cudaMemcpy (p1, p2, p3, p4);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaMemcpy in DSOs!! Dying...\n");
		exit (0);
	}
	return res;
}

cudaError_t cudaThreadSynchronize (void)
{
	int i, devid;
	cudaError_t res;

	if (real_cudaThreadSynchronize != NULL && mpitrace_on)
	{
		cudaGetDevice (&devid);

		InitializeCUDA (devid);

		Backend_Enter_Instrumentation (2);
		Probe_Cuda_ThreadBarrier_Entry ();

		res = real_cudaThreadSynchronize ();

		for (i = 0; i < devices[devid].nstreams; i++)
		{
			FlushStream (devid, i);
			CUDASynchronizeStream (devid, i);
		}

		Probe_Cuda_ThreadBarrier_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_cudaThreadSynchronize != NULL && !mpitrace_on)
	{
		res = real_cudaThreadSynchronize ();
	}
	else
	{
		fprintf (stderr, "Unable to find cudaThreadSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

cudaError_t cudaStreamSynchronize (cudaStream_t p1)
{
	int strid, devid;
	cudaError_t res;

	if (real_cudaStreamSynchronize != NULL && mpitrace_on)
	{
		cudaGetDevice (&devid);

		InitializeCUDA (devid);

		strid = SearchCUDAStream (devid, p1);
		if (strid == -1)
		{
			fprintf (stderr, "Error! Cannot determine stream index in cudaStreamSynchronize (p1=%p)\n", p1);
			exit (-1);
		}

		Backend_Enter_Instrumentation (2);
		Probe_Cuda_StreamBarrier_Entry ();

		res = real_cudaStreamSynchronize (p1);

		FlushStream (devid, strid);
		CUDASynchronizeStream (devid, strid);

		Probe_Cuda_StreamBarrier_Exit ();
		Backend_Leave_Instrumentation ();
	}
	else if (real_cudaStreamSynchronize != NULL && !mpitrace_on)
	{
		res = real_cudaStreamSynchronize (p1);
	}
	else
	{
		fprintf (stderr, "Unable to find cudaStreamSynchronize in DSOs!! Dying...\n");
		exit (0);
	}

	return res;
}

