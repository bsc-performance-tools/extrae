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

static char UNUSED rcsid[] = "$Id$";

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

	err = cudaEventRecord (devices[devid].Stream[streamid].device_reference_time,
		devices[devid].Stream[streamid].stream);
	CHECK_CU_ERROR(err, cudaEventRecord);

	err = cudaEventSynchronize (devices[devid].Stream[streamid].device_reference_time);
	CHECK_CU_ERROR(err, cudaEventSynchronize);

	devices[devid].Stream[streamid].host_reference_time = TIME;
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
		devices[devid].nstreams = 1;

		devices[devid].Stream = (struct RegisteredStreams_t*) malloc (
		  devices[devid].nstreams*sizeof(struct RegisteredStreams_t));
		if (devices[devid].Stream == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Cannot allocate information for CUDA default stream in device %d!\n", devid);
			exit (-1);
		}

		/* For timing purposes we change num of threads here instead of doing Backend_getNumberOfThreads() + CUDAdevices*/
		Backend_ChangeNumberOfThreads (Backend_getNumberOfThreads() + 1);

		/* default device stream */
		devices[devid].Stream[0].threadid = Backend_getNumberOfThreads()-1;
		devices[devid].Stream[0].stream = (cudaStream_t) 0;
		devices[devid].Stream[0].nevents = 0;

		/* Set thread name */
		{
			char _threadname[THREAD_INFO_NAME_LEN];
			char _hostname[HOST_NAME_MAX];

			if (gethostname(_hostname, HOST_NAME_MAX) == 0)
				sprintf (_threadname, "CUDA-%d.%d-%s", devid, 0, _hostname);
			else
				sprintf (_threadname, "CUDA-%d.%d-%s", devid, 0, "unknown-host");
			Extrae_set_thread_name (devices[devid].Stream[0].threadid, _threadname);
		}

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

static void Extrae_CUDA_RegisterStream (cudaStream_t stream)
{
	int i,j,devid, err; 

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

		/* Set thread name */
		{
			char _threadname[THREAD_INFO_NAME_LEN];
			char _hostname[HOST_NAME_MAX];

			if (gethostname(_hostname, HOST_NAME_MAX) == 0)
				sprintf (_threadname, "CUDA-%d.%d-%s", devid, i, _hostname);
			else
				sprintf (_threadname, "CUDA-%d.%d-%s", devid, i, "unknown-host");
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

static void Extrae_CUDA_AddEventToStream (Extrae_CUDA_Time_Type timetype, int devid,
	int streamid, unsigned event, unsigned long long type, unsigned tag,
	unsigned size)
{
	int evt_index, err;
	struct RegisteredStreams_t *ptr;

	ptr = &devices[devid].Stream[streamid];

	evt_index = ptr->nevents;

	if (evt_index < MAX_CUDA_EVENTS)
	{
		err = cudaEventRecord (ptr->ts_events[evt_index], ptr->stream);
		CHECK_CU_ERROR(err, cudaEventRecord);

		ptr->events[evt_index] = event;
		ptr->types[evt_index] = type;
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
	int i, err;
	UINT64 last_time = 0;

	/* Check whether we will fill the buffer soon (or now) */
	if (Buffer_RemainingEvents(TracingBuffer[threadid]) <= 2*devices[devid].Stream[streamid].nevents)
		Buffer_ExecuteFlushCallback (TracingBuffer[threadid]);

	/* Flush events into thread buffer */
	for (i = 0; i < devices[devid].Stream[streamid].nevents; i++)
	{
		UINT64 utmp;
		float ftmp;

		err = cudaEventSynchronize (devices[devid].Stream[streamid].ts_events[i]);
		CHECK_CU_ERROR(err, cudaEventSynchronize);

		if (devices[devid].Stream[streamid].timetype[i] == EXTRAE_CUDA_NEW_TIME)
		{
			cudaEventElapsedTime (&ftmp,
			  devices[devid].Stream[streamid].device_reference_time,
			  devices[devid].Stream[streamid].ts_events[i]);
			ftmp *= 1000000;
			utmp = devices[devid].Stream[streamid].host_reference_time + (UINT64) (ftmp);
		}
		else
			utmp = last_time;

		THREAD_TRACE_MISCEVENT (threadid, utmp,
		  devices[devid].Stream[streamid].events[i],
		  devices[devid].Stream[streamid].types[i], 0);

		if (devices[devid].Stream[streamid].events[i] == CUDAMEMCPY_GPU_EV)
			if (devices[devid].Stream[streamid].tag[i] > 0)
				THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, utmp,
				 (devices[devid].Stream[streamid].types[i]==EVT_END)?USER_RECV_EV:USER_SEND_EV,
				 TASKID,
				 devices[devid].Stream[streamid].size[i],
				 devices[devid].Stream[streamid].tag[i],
				 devices[devid].Stream[streamid].tag[i]);

		last_time = utmp;
	}
	devices[devid].Stream[streamid].nevents = 0;
}

/****************************************************************************/
/* CUDA INSTRUMENTATION                                                     */
/****************************************************************************/

static int _cudaLaunch_stream = 0;

void Extrae_cudaLaunch_Enter (int devid, cudaLaunch_v3020_params* p)
{
	Backend_Enter_Instrumentation (2);
	Probe_Cuda_Launch_Entry ((UINT64) p->entry);
	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream, CUDAKERNEL_GPU_EV, (UINT64) p->entry, 0, 0);
}

void Extrae_cudaLaunch_Exit (int devid, cudaLaunch_v3020_params* p)
{
	UNREFERENCED_PARAMETER(p);

	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream, CUDAKERNEL_GPU_EV, EVT_END, 0, 0);
	Probe_Cuda_Launch_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaConfigureCall_Enter (int devid, cudaConfigureCall_v3020_params* p)
{
	int strid;

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_ConfigureCall_Entry ();
	strid = Extrae_CUDA_SearchStream (devid, p->stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaConfigureCall (p->stream=%p)\n", p->stream);
		exit (-1);
	}
	_cudaLaunch_stream = strid;
	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid, CUDACONFIGKERNEL_GPU_EV, EVT_BEGIN, 0, 0);
}

void Extrae_cudaConfigureCall_Exit (int devid, cudaConfigureCall_v3020_params* p)
{
	UNREFERENCED_PARAMETER(p);
	UNREFERENCED_PARAMETER(devid);

	Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, _cudaLaunch_stream, CUDACONFIGKERNEL_GPU_EV, EVT_END, 0, 0);
	Probe_Cuda_ConfigureCall_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaThreadSynchronize_Enter (int devid)
{
	UNREFERENCED_PARAMETER(devid);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_ThreadBarrier_Entry ();
}

void Extrae_cudaThreadSynchronize_Exit (int devid)
{
	int i;

	for (i = 0; i < devices[devid].nstreams; i++)
	{
		Extrae_CUDA_FlushStream (devid, i);
		Extrae_CUDA_SynchronizeStream (devid, i);
	}

	Probe_Cuda_ThreadBarrier_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaStreamCreate_Exit (int devid, cudaStreamCreate_v3020_params* p)
{
	UNREFERENCED_PARAMETER(devid);

	Extrae_CUDA_RegisterStream (*(p->pStream));
}

void Extrae_cudaStreamSynchronize_Enter (int devid, cudaStreamSynchronize_v3020_params* p)
{
	UNREFERENCED_PARAMETER(p);
	UNREFERENCED_PARAMETER(devid);

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_StreamBarrier_Entry ();
}

void Extrae_cudaStreamSynchronize_Exit (int devid, cudaStreamSynchronize_v3020_params* p)
{
	int strid;

	strid = Extrae_CUDA_SearchStream (devid, p->stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in cudaStreamSynchronize\n");
		exit (-1);
	}
	Extrae_CUDA_FlushStream (devid, strid);
	Extrae_CUDA_SynchronizeStream (devid, strid);
	Probe_Cuda_StreamBarrier_Exit ();
	Backend_Leave_Instrumentation ();
}

void Extrae_cudaMemcpy_Enter (int devid, cudaMemcpy_v3020_params *p)
{
	unsigned tag;

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_Memcpy_Entry (p->count);

	tag = Extrae_CUDA_tag_generator();

	if (p->kind == cudaMemcpyHostToDevice || p->kind == cudaMemcpyHostToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, p->count, tag, tag);
	}

	if (p->kind == cudaMemcpyHostToDevice || p->kind == cudaMemcpyHostToHost)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0, CUDAMEMCPY_GPU_EV, p->count, 0, 0);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0, CUDAMEMCPY_GPU_EV, p->count, tag, p->count);
}

void Extrae_cudaMemcpy_Exit (int devid, cudaMemcpy_v3020_params *p)
{
	int i;
	unsigned tag;

	tag = Extrae_CUDA_tag_generator();

	if (p->kind == cudaMemcpyHostToDevice || p->kind == cudaMemcpyDeviceToDevice)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0, CUDAMEMCPY_GPU_EV, EVT_END, tag, p->count);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, 0, CUDAMEMCPY_GPU_EV, EVT_END, 0, 0);

	for (i = 0; i < devices[devid].nstreams; i++)
	{
		Extrae_CUDA_FlushStream (devid, i);
		Extrae_CUDA_SynchronizeStream (devid, i);
	}

	Probe_Cuda_Memcpy_Exit ();

	if (p->kind == cudaMemcpyDeviceToHost || p->kind == cudaMemcpyHostToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, p->count, tag, tag);
	}

	Backend_Leave_Instrumentation ();
}

void Extrae_cudaMemcpyAsync_Enter (int devid, cudaMemcpyAsync_v3020_params *p)
{
	int strid;
	unsigned tag;

	Backend_Enter_Instrumentation (2);
	Probe_Cuda_Memcpy_Entry (p->count);

	tag = Extrae_CUDA_tag_generator();

	if (p->kind == cudaMemcpyHostToDevice || p->kind == cudaMemcpyHostToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, p->count, tag, tag);
	}

	strid = Extrae_CUDA_SearchStream (devid, p->stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_cudaMemcpyAsync_Enter\n");
		exit (-1);
	}

	if (p->kind == cudaMemcpyHostToDevice || p->kind == cudaMemcpyHostToHost)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid, CUDAMEMCPY_GPU_EV, p->count, 0, 0);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid, CUDAMEMCPY_GPU_EV, p->count, tag, p->count);
}

void Extrae_cudaMemcpyAsync_Exit (int devid, cudaMemcpyAsync_v3020_params *p)
{
	int strid;
	unsigned tag;

	tag = Extrae_CUDA_tag_generator();

	strid = Extrae_CUDA_SearchStream (devid, p->stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_cudaMemcpyAsync_Enter\n");
		exit (-1);
	}

	if (p->kind == cudaMemcpyHostToDevice || p->kind == cudaMemcpyDeviceToDevice)
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid, CUDAMEMCPY_GPU_EV, EVT_END, tag, p->count);
	else
		Extrae_CUDA_AddEventToStream (EXTRAE_CUDA_NEW_TIME, devid, strid, CUDAMEMCPY_GPU_EV, EVT_END, 0, 0);

	Probe_Cuda_Memcpy_Exit ();

	if (p->kind == cudaMemcpyDeviceToHost || p->kind == cudaMemcpyHostToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, p->count, tag, tag);
	}

	Backend_Leave_Instrumentation ();
}

