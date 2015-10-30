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
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.3/src/tracer/wrappers/CUDA/cuda_wrapper.c $
 | @last_commit: $Date: 2013-04-30 15:15:24 +0200 (dt, 30 abr 2013) $
 | @version:     $Revision: 1696 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: cuda_wrapper.c 1696 2013-04-30 13:15:24Z harald $";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include "opencl_common.h"
#include "opencl_wrapper.h"
#include "taskid.h"
#include "threadinfo.h"
#include "wrapper.h"

#define MAX_OPENCL_EVENTS 32768

#if defined(__APPLE__)
# define HOST_NAME_MAX 512
#endif

typedef struct RegisteredCommandQueue_st
{
	cl_command_queue queue;
	int isOutOfOrder;
	
	UINT64 host_reference_time;
	cl_ulong device_reference_time;
	int threadid; /* In Paraver sense */

	unsigned nevents;
	cl_event ocl_event[MAX_OPENCL_EVENTS];
	unsigned prv_event[MAX_OPENCL_EVENTS];
	cl_kernel k_event[MAX_OPENCL_EVENTS]; /* For those events that are accompanied by kernels */
	size_t size[MAX_OPENCL_EVENTS]; /* For those events that are accompanied by a size */
} RegisteredCommandQueue_t;

static unsigned nCommandQueues = 0;
static RegisteredCommandQueue_t *CommandQueues;

static unsigned nKernels = 0;
typedef struct
{
	char *KernelName;
} AnnotatedKernel_st;
static AnnotatedKernel_st *Kernels;
//static cl_kernel *Kernels;

static unsigned __last_tag = 0x0C31; /* Fixed tag */
unsigned Extrae_OpenCL_tag_generator (void)
{
	return __last_tag;
}

static int Extrae_OpenCL_lookForOpenCLQueue (cl_command_queue q, unsigned *position)
{
	unsigned u;

	for (u = 0; u < nCommandQueues; u++)
		if (CommandQueues[u].queue == q)
		{
			if (position != NULL)
				*position = u;
			return TRUE;
		}

	return FALSE;
}

int Extrae_OpenCL_Queue_OoO (cl_command_queue q)
{
	unsigned idx;
	if (Extrae_OpenCL_lookForOpenCLQueue (q, &idx))
		return CommandQueues[idx].isOutOfOrder;
	else
		return FALSE;
}

static int Extrae_OpenCL_lookForKernelName (const char *kernel_name, unsigned *position)
{
	unsigned u;

	if (position)
		*position = 0;
	for (u = 0; u < nKernels; u++)
		if (!strcmp (kernel_name, Kernels[u].KernelName))
		{
			if (position != NULL)
				*position = u;
			return TRUE;
		}
	return FALSE;
}

void Extrae_OpenCL_clCreateCommandQueue (cl_command_queue queue,
	cl_device_id device, cl_command_queue_properties properties)
{
	if (!Extrae_OpenCL_lookForOpenCLQueue (queue, NULL))
	{
		cl_int err;
		char _threadname[THREAD_INFO_NAME_LEN];
		char _hostname[HOST_NAME_MAX];
		char *_device_type;
		int prev_threadid, found, idx;
		cl_device_type device_type;
		cl_event event;

		idx = nCommandQueues;
		CommandQueues = (RegisteredCommandQueue_t*) realloc (
			CommandQueues,
			sizeof(RegisteredCommandQueue_t)*(nCommandQueues+1));
		if (CommandQueues == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Fatal error! Failed to allocate memory for OpenCL Command Queues\n");
			exit (-1);
		}

		CommandQueues[idx].queue = queue;
		CommandQueues[idx].isOutOfOrder =
			(properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;

		err = clGetDeviceInfo (device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
		if (err == CL_SUCCESS)
		{
			if (device_type  == CL_DEVICE_TYPE_GPU)
				_device_type = "GPU";
			else if (device_type == CL_DEVICE_TYPE_CPU)
				_device_type = "CPU";
			else
				_device_type = "Other";
		}
		else
			_device_type = "Unknown";

		/* Was the thread created before (i.e. did we executed a cudadevicereset?) */
		if (gethostname(_hostname, HOST_NAME_MAX) == 0)
			sprintf (_threadname, "OpenCL-%s-CQ%d-%s", _device_type, 1, _hostname);
		else
			sprintf (_threadname, "OpenCL-%s-CQ%d-%s", _device_type, 1, "unknown-host");

		prev_threadid = Extrae_search_thread_name (_threadname, &found);

		if (found)
		{
			/* If thread name existed, reuse its thread id */
			CommandQueues[idx].threadid = prev_threadid;
		}
		else
		{
			/* For timing purposes we change num of threads here instead of doing Backend_getNumberOfThreads() + CUDAdevices*/
			Backend_ChangeNumberOfThreads (Backend_getNumberOfThreads() + 1);
			CommandQueues[idx].threadid = Backend_getNumberOfThreads()-1;

			/* Set thread name */
			Extrae_set_thread_name (CommandQueues[idx].threadid, _threadname);
		}

		CommandQueues[idx].nevents = 0;

#ifdef CL_VERSION_1_2
		err = clEnqueueBarrierWithWaitList (queue, 0, NULL, &event);
#else
		err = clEnqueueBarrier (queue);
		if (err == CL_SUCCESS)
			err = clEnqueueMarker (queue, &event);
#endif
		CommandQueues[idx].host_reference_time = TIME;

		if (err == CL_SUCCESS)
		{
			err = clFinish(queue);
			if (err != CL_SUCCESS)
			{
				fprintf (stderr, PACKAGE_NAME": Error in clFinish (error = %d)! Dying...\n", err);
				exit (-1);
			}

			err = clGetEventProfilingInfo (event, CL_PROFILING_COMMAND_SUBMIT,
				sizeof(cl_ulong), &(CommandQueues[idx].device_reference_time),
				NULL);
			if (err != CL_SUCCESS)
			{
				fprintf (stderr, PACKAGE_NAME": Error in clGetEventProfilingInfo (error = %d)! Dying...\n", err);
				exit (-1);
			}
		}
		else
		{
			fprintf (stderr, PACKAGE_NAME": Error while looking for clock references in host & accelerator\n");
			exit (-1);
		}

		nCommandQueues++;
	}
}

void Extrae_OpenCL_addEventToQueue (cl_command_queue queue, cl_event ocl_evt, 
	unsigned prv_evt)
{
	unsigned idx, idx2;

	if (!Extrae_OpenCL_lookForOpenCLQueue (queue, &idx))
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! Cannot find OpenCL command queue!\n");
		exit (-1);
	}
	if (CommandQueues[idx].nevents >= MAX_OPENCL_EVENTS)
	{
		fprintf (stderr, PACKAGE_NAME": Error! OpenCL tracing buffer overrun! Execute clFinish more frequently or ncrease MAX_OPENCL_EVENTS in "__FILE__);
		return;
	}

	idx2 = CommandQueues[idx].nevents;
	CommandQueues[idx].ocl_event[idx2] = ocl_evt;
	CommandQueues[idx].prv_event[idx2] = prv_evt;
	CommandQueues[idx].k_event[idx2] = NULL;
	CommandQueues[idx].size[idx2] = 0;
	CommandQueues[idx].nevents++;
	Extrae_clRetainEvent_real (ocl_evt);
}

void Extrae_OpenCL_addEventToQueueWithSize (cl_command_queue queue, cl_event ocl_evt, 
	unsigned prv_evt, size_t size)
{
	unsigned idx, idx2;

	if (!Extrae_OpenCL_lookForOpenCLQueue (queue, &idx))
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! Cannot find OpenCL command queue!\n");
		exit (-1);
	}
	if (CommandQueues[idx].nevents >= MAX_OPENCL_EVENTS)
	{
		fprintf (stderr, PACKAGE_NAME": Error! OpenCL tracing buffer overrun! Execute clFinish more frequently or ncrease MAX_OPENCL_EVENTS in "__FILE__);
		return;
	}

	idx2 = CommandQueues[idx].nevents;
	CommandQueues[idx].ocl_event[idx2] = ocl_evt;
	CommandQueues[idx].prv_event[idx2] = prv_evt;
	CommandQueues[idx].k_event[idx2] = NULL;
	CommandQueues[idx].size[idx2] = size;
	CommandQueues[idx].nevents++;
	Extrae_clRetainEvent_real (ocl_evt);
}

void Extrae_OpenCL_addEventToQueueWithKernel (cl_command_queue queue,
	cl_event ocl_evt, unsigned prv_evt, cl_kernel k)
{
	unsigned idx, idx2;

	if (!Extrae_OpenCL_lookForOpenCLQueue (queue, &idx))
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! Cannot find OpenCL command queue!\n");
		exit (-1);
	}
	if (CommandQueues[idx].nevents >= MAX_OPENCL_EVENTS)
	{
		fprintf (stderr, PACKAGE_NAME": Error! OpenCL tracing buffer overrun! Execute clFinish more frequently or ncrease MAX_OPENCL_EVENTS in "__FILE__);
		return;
	}

	idx2 = CommandQueues[idx].nevents;
	CommandQueues[idx].ocl_event[idx2] = ocl_evt;
	CommandQueues[idx].prv_event[idx2] = prv_evt;
	CommandQueues[idx].k_event[idx2] = k;
	CommandQueues[idx].size[idx2] = 0;
	CommandQueues[idx].nevents++;
	Extrae_clRetainEvent_real (ocl_evt);
}

static void Extrae_OpenCL_comm_at_OpenCL (RegisteredCommandQueue_t * cq, unsigned pos,
	unsigned long long t, int entry)
{
	if (entry)
	{
		if (cq->prv_event[pos] == OPENCL_CLENQUEUEREADBUFFER_ACC_EV ||
		    cq->prv_event[pos] == OPENCL_CLENQUEUEREADBUFFER_ASYNC_ACC_EV)
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(cq->threadid, t,
			  USER_SEND_EV, TASKID, cq->size[pos],
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());
		}
		else if (cq->prv_event[pos] == OPENCL_CLENQUEUEREADBUFFERRECT_ACC_EV ||
		    cq->prv_event[pos] == OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_ACC_EV)
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(cq->threadid, t,
			  USER_SEND_EV, TASKID, 0,
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());
		}
	}
	else
	{
		if (cq->prv_event[pos] == OPENCL_CLENQUEUENDRANGEKERNEL_ACC_EV ||
		    cq->prv_event[pos] == OPENCL_CLENQUEUETASK_ACC_EV ||
		    cq->prv_event[pos] == OPENCL_CLENQUEUENATIVEKERNEL_ACC_EV)
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(cq->threadid, t,
			  USER_RECV_EV, TASKID, 0,
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());
		}
		else if (cq->prv_event[pos] == OPENCL_CLENQUEUEWRITEBUFFER_ACC_EV ||
		    cq->prv_event[pos] == OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_ACC_EV)
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(cq->threadid, t,
			  USER_RECV_EV, TASKID, cq->size[pos],
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());
		}
		else if (cq->prv_event[pos] == OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV ||
		    cq->prv_event[pos] == OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_ACC_EV)
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(cq->threadid, t,
			  USER_RECV_EV, TASKID, 0,
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());
		}
	}
}


static void Extrae_OpenCL_real_clQueueFlush (unsigned idx, int addFinish)
{
	unsigned u;
	int threadid = CommandQueues[idx].threadid;
	unsigned remainingevts = Buffer_RemainingEvents(TracingBuffer[threadid]);
	cl_ulong last_time = 0;

	cl_long delta_time = ((cl_long) CommandQueues[idx].host_reference_time) - 
		((cl_long) CommandQueues[idx].device_reference_time);

	/* Check whether we will fill the buffer soon (or now) */
	if (remainingevts <= 2*CommandQueues[idx].nevents+2)
		Buffer_ExecuteFlushCallback (TracingBuffer[threadid]);

	/* Flush events into thread buffer */
	for (u = 0; u < CommandQueues[idx].nevents; u++)
	{
		cl_int err;
		cl_ulong utmp;
		cl_event evt = CommandQueues[idx].ocl_event[u];

		err = clGetEventProfilingInfo (evt, CL_PROFILING_COMMAND_START,
			sizeof(utmp), &utmp, NULL);
		if (err != CL_SUCCESS)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Cannot obtain OpenCL profiling info!\n");
			continue;
		}

		utmp = utmp + delta_time; /* Correct timing between Host & Accel */

		if (CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEREADBUFFER_ACC_EV ||
		    CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEWRITEBUFFER_ACC_EV ||
		    CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEREADBUFFERRECT_ACC_EV ||
		    CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV ||
		    CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEREADBUFFER_ASYNC_ACC_EV ||
		    CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_ACC_EV ||
		    CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_ACC_EV ||
		    CommandQueues[idx].prv_event[u] == OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_ACC_EV)
		{
			THREAD_TRACE_MISCEVENT (threadid, utmp,
			  CommandQueues[idx].prv_event[u], EVT_BEGIN,
			  CommandQueues[idx].size[u]);
		}
		else
		{
			if (CommandQueues[idx].k_event[u] != NULL)
			{
				unsigned val;
				Extrae_OpenCL_annotateKernelName (CommandQueues[idx].k_event[u], &val);
				THREAD_TRACE_MISCEVENT (threadid, utmp,
				  CommandQueues[idx].prv_event[u], EVT_BEGIN, val+1);
			}
			else
			{
				THREAD_TRACE_MISCEVENT (threadid, utmp,
				  CommandQueues[idx].prv_event[u], EVT_BEGIN, 0);
			}
		}

		Extrae_OpenCL_comm_at_OpenCL (&CommandQueues[idx], u, utmp,
		  TRUE);

		err = clGetEventProfilingInfo (evt, CL_PROFILING_COMMAND_END,
			sizeof(utmp), &utmp, NULL);
		if (err != CL_SUCCESS)
		{
			fprintf (stderr, PACKAGE_NAME": Error! Cannot obtain OpenCL profiling info!\n");
			continue;
		}

		utmp = utmp + delta_time; /* Correct timing between Host & Accel */

		THREAD_TRACE_MISCEVENT (threadid, utmp,
		  CommandQueues[idx].prv_event[u], EVT_END, 0);

		Extrae_OpenCL_comm_at_OpenCL (&CommandQueues[idx], u, utmp, FALSE);

		Extrae_clReleaseEvent_real (evt);

		last_time = utmp;
	}

	if (addFinish && CommandQueues[idx].nevents > 0)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID,
			0, Extrae_OpenCL_tag_generator(), OPENCL_CLFINISH_EV);
		THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, last_time,
		  USER_SEND_EV, TASKID, 0, Extrae_OpenCL_tag_generator(), OPENCL_CLFINISH_EV);
	}

	CommandQueues[idx].nevents = 0;
}

void Extrae_OpenCL_clQueueFlush (cl_command_queue queue, int addFinish)
{
	unsigned idx;

	if (!Extrae_OpenCL_lookForOpenCLQueue (queue, &idx))
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! Cannot find OpenCL command queue!\n");
		exit (-1);
	}

	Extrae_OpenCL_real_clQueueFlush (idx, addFinish);
}

void Extrae_OpenCL_clQueueFlush_All (void)
{
	unsigned u;

	for (u = 0; u < nCommandQueues; u++)
		Extrae_OpenCL_real_clQueueFlush (u, FALSE);
}

static char *Extrae_OpenCL_getKernelName (cl_kernel k)
{
	cl_int ret;
	size_t len;

	ret = clGetKernelInfo (k, CL_KERNEL_FUNCTION_NAME, 0, NULL, &len);
	if (CL_SUCCESS == ret)
	{
		char name[len+1];
		ret = clGetKernelInfo (k, CL_KERNEL_FUNCTION_NAME, len, name, NULL);
		if (CL_SUCCESS == ret)
			return strdup (name);
	}

	return "unnamed";
}

void Extrae_OpenCL_annotateKernelName (cl_kernel k, unsigned *pos)
{
	/* Add a new entry if the kernel name does not exist */
	char *kname = Extrae_OpenCL_getKernelName (k);

	if (!Extrae_OpenCL_lookForKernelName (kname, pos))
	{
		unsigned long long v;
	
		Kernels = (AnnotatedKernel_st*) realloc (Kernels,
		  sizeof(AnnotatedKernel_st)*(nKernels+1));

		if (Kernels == NULL)
		{
			fprintf (stderr, PACKAGE_NAME": Fatal error! Failed to allocate memory for OpenCL Kernels\n");
			exit (-1);
		}

		Kernels[nKernels].KernelName = strdup (kname);
		*pos = nKernels;
		v = nKernels+1;

		Extrae_AddTypeValuesEntryToLocalSYM ('D', OPENCL_KERNEL_NAME_EV,
			"OpenCL kernel name", 'd', 1, &v, &kname);

		nKernels++;
	}

	/* Free allocated kernel name */
	free (kname);
}

