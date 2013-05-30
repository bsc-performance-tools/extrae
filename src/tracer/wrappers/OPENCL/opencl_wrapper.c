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
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <CL/cl.h>
#include "wrapper.h"
#include "opencl_probe.h"

static cl_mem (*real_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int *) = NULL;
static cl_command_queue (*real_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = NULL;
static cl_context (*real_clCreateContext)(const cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *) = NULL;
static cl_context (*real_clCreateContextFromType)(const cl_context_properties *, cl_device_type, void *, void *, cl_int *) = NULL;
static cl_kernel (*real_clCreateKernel)(cl_program, const char *, cl_int *) = NULL;
static cl_int (*real_clCreateKernelsInProgram)(cl_program, cl_uint, cl_kernel *, cl_uint *) = NULL;
static cl_int (*real_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *) = NULL;
static cl_program (*real_clCreateProgramWithSource)(cl_context, cl_uint, const char **,	const size_t *, cl_int *) = NULL;
static cl_program (*real_clCreateProgramWithBinary)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *) = NULL;
static cl_program (*real_clCreateProgramWithBuiltInKernels)(cl_context, cl_uint, const cl_device_id *, const char *, cl_int *) = NULL;
static cl_mem (*real_clCreateSubBuffer)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void *, cl_int *) = NULL;
static cl_int (*real_clEnqueueFillBuffer)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueCopyBufferRect)(cl_command_queue, cl_mem, cl_mem,	const size_t *, const size_t *, const size_t *, size_t, size_t,	size_t, size_t, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint,	const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueTask)(cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueNativeKernel)(cl_command_queue, void *, void *, size_t, cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueReadBufferRect)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueWriteBufferRect)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clBuildProgram)(cl_program, cl_uint, const cl_device_id *,	const char *, void *, void *) = NULL;
static cl_int (*real_clCompileProgram)(cl_program, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *, const char **, void *, void *) = NULL;
static cl_program (*real_clLinkProgram)(cl_context, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *, void *, void *, cl_int *) = NULL;
static cl_int (*real_clFinish)(cl_command_queue) = NULL;
static cl_int (*real_clFlush)(cl_command_queue) = NULL;
static cl_int (*real_clWaitForEvents)(cl_uint, const cl_event *el) = NULL;
static cl_int (*real_clEnqueueMarkerWithWaitList)(cl_command_queue, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueBarrierWithWaitList)(cl_command_queue, cl_uint, const cl_event *, cl_event *) = NULL;

void Extrae_OpenCL_init (unsigned rank)
{
	real_clCreateBuffer = (cl_mem(*)(cl_context, cl_mem_flags, size_t, void*, cl_int *))
		dlsym (RTLD_NEXT, "clCreateBuffer");
	if (real_clCreateBuffer == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateBuffer in DSOs!!\n");

	real_clCreateCommandQueue = (cl_command_queue(*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*))
		dlsym (RTLD_NEXT, "clCreateCommandQueue");
	if (real_clCreateCommandQueue == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateCommandQueue in DSOs!!\n");

	real_clCreateContext = (cl_context(*)(const cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *))
		dlsym (RTLD_NEXT, "clCreateContext");
	if (real_clCreateContext == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateContext in DSOs!!\n");

	real_clCreateContextFromType = (cl_context(*)(const cl_context_properties *, cl_device_type, void *, void *, cl_int *))
		dlsym (RTLD_NEXT, "clCreateContextFromType");
	if (real_clCreateContextFromType == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateContextFromType in DSOs!!\n");

	real_clCreateKernel = (cl_kernel(*)(cl_program, const char *, cl_int *))
		dlsym (RTLD_NEXT, "clCreateKernel");
	if (real_clCreateKernel == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateKernel in DSOs!!\n");

	real_clCreateKernelsInProgram = (cl_int(*)(cl_program, cl_uint, cl_kernel *, cl_uint *))
		dlsym (RTLD_NEXT, "clCreateKernelsInProgram");
	if (real_clCreateKernelsInProgram == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateKernelsInProgram in DSOs!!\n");

	real_clSetKernelArg = (cl_int(*)(cl_kernel, cl_uint, size_t, const void *))
		dlsym (RTLD_NEXT, "clSetKernelArg");
	if (real_clSetKernelArg == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clSetKernelArg in DSOs!!\n");

	real_clCreateProgramWithSource = (cl_program(*)(cl_context, cl_uint, const char **, const size_t *, cl_int *))
		dlsym (RTLD_NEXT, "clCreateProgramWithSource");
	if (real_clCreateProgramWithSource == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateProgramWithSource in DSOs!!\n");

	real_clCreateProgramWithBinary = (cl_program(*)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *))
		dlsym (RTLD_NEXT, "clCreateProgramWithBinary");
	if (real_clCreateProgramWithBinary == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateProgramWithBinary in DSOs!!\n");

	real_clCreateProgramWithBuiltInKernels = (cl_program(*)(cl_context, cl_uint, const cl_device_id *, const char *, cl_int *))
		dlsym (RTLD_NEXT, "clCreateProgramWithBuiltInKernels");
	if (real_clCreateProgramWithBuiltInKernels == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateProgramWithBuiltInKernels in DSOs!!\n");

	real_clCreateSubBuffer = (cl_mem(*)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void *, cl_int *))
		dlsym (RTLD_NEXT, "clCreateSubBuffer");
	if (real_clCreateSubBuffer == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCreateSubBuffer in DSOs!!\n");

	real_clEnqueueFillBuffer = (cl_int(*)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueFillBuffer");
	if (real_clEnqueueFillBuffer == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueFillBuffer in DSOs!!\n");

	real_clEnqueueCopyBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueCopyBuffer");
	if (real_clEnqueueCopyBuffer == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueCopyBuffer in DSOs!!\n");

	real_clEnqueueCopyBufferRect = (cl_int(*)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueCopyBufferRect");
	if (real_clEnqueueCopyBufferRect == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueCopyBufferRect in DSOs!!\n");

	real_clEnqueueNDRangeKernel = (cl_int(*)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueNDRangeKernel");
	if (real_clEnqueueNDRangeKernel == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueNDRangeKernel in DSOs!!\n");

	real_clEnqueueTask = (cl_int(*)(cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueTask");
	if (real_clEnqueueTask == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueTask in DSOs!!\n");

	real_clEnqueueNativeKernel = (cl_int(*)(cl_command_queue, void *, void *, size_t, cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueNativeKernel");
	if (real_clEnqueueNativeKernel == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueNativeKernel in DSOs!!\n");

	real_clEnqueueReadBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueReadBuffer");
	if (real_clEnqueueReadBuffer == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueReadBuffer in DSOs!!\n");

	real_clEnqueueReadBufferRect = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueReadBufferRect");
	if (real_clEnqueueReadBufferRect == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueReadBufferRect in DSOs!!\n");
	real_clEnqueueWriteBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueWriteBuffer");
	if (real_clEnqueueWriteBuffer == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueWriteBuffer in DSOs!!\n");

	real_clEnqueueWriteBufferRect = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueWriteBufferRect");
	if (real_clEnqueueWriteBufferRect == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueWriteBufferRect in DSOs!!\n");

	real_clBuildProgram = (cl_int(*)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *))
		dlsym (RTLD_NEXT, "clBuildProgram");
	if (real_clBuildProgram == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clBuildProgram in DSOs!!\n");

	real_clCompileProgram = (cl_int(*)(cl_program, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *, const char **, void *, void *))
		dlsym (RTLD_NEXT, "clCompileProgram");
	if (real_clCompileProgram == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clCompileProgram in DSOs!!\n");

	real_clLinkProgram = (cl_program(*)(cl_context, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *, void *, void *, cl_int *))
		dlsym (RTLD_NEXT, "clLinkProgram");
	if (real_clLinkProgram == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clLinkProgram in DSOs!!\n");

	real_clFinish = (cl_int(*)(cl_command_queue))
		dlsym (RTLD_NEXT, "clFinish");
	if (real_clFinish == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clFinish in DSOs!!\n");

	real_clFlush = (cl_int(*)(cl_command_queue))
		dlsym (RTLD_NEXT, "clFlush");
	if (real_clFlush == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clFlush in DSOs!!\n");

	real_clWaitForEvents = (cl_int(*)(cl_uint, const cl_event *el))
		dlsym (RTLD_NEXT, "clWaitForEvents");
	if (real_clWaitForEvents == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clWaitForEvents in DSOs!!\n");

	real_clEnqueueMarkerWithWaitList = (cl_int(*)(cl_command_queue, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueMarkerWithWaitList");
	if (real_clEnqueueMarkerWithWaitList == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueMarkerWithWaitList");

	real_clEnqueueBarrierWithWaitList = (cl_int(*)(cl_command_queue, cl_uint, const cl_event *, cl_event *))
		dlsym (RTLD_NEXT, "clEnqueueBarrierWithWaitList");
	if (real_clEnqueueBarrierWithWaitList == NULL && rank == 0)
		fprintf (stderr, PACKAGE_NAME": Unable to find clEnqueueBarrierWithWaitList");

}

cl_mem clCreateBuffer (cl_context c, cl_mem_flags m, size_t s, void *p, 
	cl_int *e)
{
	cl_mem r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateBuffer (real at %p)\n", real_clCreateBuffer);
#endif

	if (mpitrace_on && real_clCreateBuffer != NULL)
	{
		Extrae_Probe_clCreateBuffer_Enter();
		r = real_clCreateBuffer (c, m, s, p, e);
		Extrae_Probe_clCreateBuffer_Exit();
	}
	else if (!mpitrace_on && real_clCreateBuffer != NULL)
	{
		r = real_clCreateBuffer (c, m, s, p, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME" Fatal Error! clCreateBuffer was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_command_queue clCreateCommandQueue (cl_context c, cl_device_id d,
	cl_command_queue_properties p, cl_int *e)
{
	cl_command_queue r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateCommandQueue (real at %p)\n", real_clCreateCommandQueue);
#endif

	/* Force profiling! */
	p |= CL_QUEUE_PROFILING_ENABLE;

	if (mpitrace_on && real_clCreateCommandQueue != NULL)
	{
		Extrae_Probe_clCreateCommandQueue_Enter ();
		r = real_clCreateCommandQueue (c, d, p, e);
		Extrae_Probe_clCreateCommandQueue_Exit ();
	}
	else if (!mpitrace_on && real_clCreateCommandQueue != NULL)
	{
		r = real_clCreateCommandQueue (c, d, p, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCraeteCommandQueue was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_context clCreateContext (const cl_context_properties *p, cl_uint n, 
	const cl_device_id *d,
	void (*pfn)(const char *, const void *, size_t, void *),
	void *udata, cl_int *e)
{
	cl_context r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateContext (real at %p)\n", real_clCreateContext);
#endif

	if (mpitrace_on && real_clCreateContext != NULL)
	{
		Extrae_Probe_clCreateContext_Enter();
		r = real_clCreateContext (p, n, d, pfn, udata, e);
		Extrae_Probe_clCreateContext_Exit();
	}
	else if (!mpitrace_on && real_clCreateContext != NULL)
	{
		r = real_clCreateContext (p, n, d, pfn, udata, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateContext was not hooked!\n");
		exit (-1);
	} 

	return r;
}

cl_context clCreateContextFromType (const cl_context_properties *p,
	cl_device_type dt,
	void (*pfn)(const char *, const void *, size_t, void *),
	void *udata, cl_int *e)
{
	cl_context r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateContextFromType (real at %p)\n", real_clCreateContextFromType);
#endif

	if (mpitrace_on && real_clCreateContextFromType != NULL)
	{
		Extrae_Probe_clCreateContextFromType_Enter ();
		r = real_clCreateContextFromType (p, dt, pfn, udata, e);
		Extrae_Probe_clCreateContextFromType_Exit ();
	}
	else if (!mpitrace_on && real_clCreateContextFromType != NULL)
	{
		r = real_clCreateContextFromType (p, dt, pfn, udata, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateContextFromType was not hooked\n");
		exit (-1);
	}

	return r;
}

cl_mem clCreateSubBuffer (cl_mem m, cl_mem_flags mf, cl_buffer_create_type bct,
	const void *b, cl_int *e)
{
	cl_mem r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateSubBuffer (real at %p)\n", real_clCreateSubBuffer);
#endif

	if (mpitrace_on && real_clCreateSubBuffer != NULL)
	{
		Extrae_Probe_clCreateSubBuffer_Enter ();
		r = real_clCreateSubBuffer (m, mf, bct, b, e);
		Extrae_Probe_clCreateSubBuffer_Exit ();
	}
	else if (!mpitrace_on && real_clCreateSubBuffer != NULL)
	{
		r = real_clCreateSubBuffer (m, mf, bct, b, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateSubBuffer was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_kernel clCreateKernel (cl_program p, const char *k, cl_int *e)
{
	cl_kernel r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateKernel (%s) (real at %p)\n", k, real_clCreateKernel);
#endif

	if (mpitrace_on && real_clCreateKernel != NULL)
	{
		Extrae_Probe_clCreateKernel_Enter ();
		r = real_clCreateKernel (p, k, e);
		Extrae_Probe_clCreateKernel_Exit ();
	}
	else if (!mpitrace_on && real_clCreateKernel != NULL)
	{
		r = real_clCreateKernel (p, k, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateKernel was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clCreateKernelsInProgram (cl_program p, cl_uint n, cl_kernel *ks, cl_uint *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateKernelsInProgram (%p)\n", real_clCreateKernelsInProgram);
#endif

	if (mpitrace_on && real_clCreateKernelsInProgram != NULL)
	{
		Extrae_Probe_clCreateKernelsInProgram_Enter ();
		r = real_clCreateKernelsInProgram (p, n, ks, e);
		Extrae_Probe_clCreateKernelsInProgram_Exit ();
	}
	else if (!mpitrace_on && real_clCreateKernelsInProgram != NULL)
	{
		r = real_clCreateKernelsInProgram (p, n, ks, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateKernelsInProgram was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clSetKernelArg (cl_kernel k, cl_uint a, size_t as, const void *av)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clSetKernelArg (real at %p)\n", real_clSetKernelArg);
#endif

	if (mpitrace_on && real_clSetKernelArg != NULL)
	{
		Extrae_Probe_clSetKernelArg_Enter ();
		r = real_clSetKernelArg (k, a, as, av);
		Extrae_Probe_clSetKernelArg_Exit ();
	}
	else if (!mpitrace_on && real_clSetKernelArg != NULL)
	{
		r = real_clSetKernelArg (k, a, as, av);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clSetKernelArg was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_program clCreateProgramWithSource (cl_context c, cl_uint u, const char **s,
	const size_t *l, cl_int *e)
{
	cl_program r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateProgramWithSource (real at %p)\n", real_clCreateProgramWithSource);
#endif

	if (mpitrace_on && real_clCreateProgramWithSource != NULL)
	{
		Extrae_Probe_clCreateProgramWithSource_Enter ();
		r = real_clCreateProgramWithSource (c, u, s, l, e);
		Extrae_Probe_clCreateProgramWithSource_Exit ();
	}
	else if (!mpitrace_on && real_clCreateProgramWithSource != NULL)
	{
		r = real_clCreateProgramWithSource (c, u, s, l, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateProgramWithSource was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_program clCreateProgramWithBinary (cl_context c, cl_uint n,
	const cl_device_id *dl, const size_t *l, const unsigned char **b,
	cl_int *bs, cl_int *e)
{
	cl_program r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateProgramWithBinary (real at %p)\n", real_clCreateProgramWithBinary);
#endif

	if (mpitrace_on && real_clCreateProgramWithBinary != NULL)
	{
		Extrae_Probe_clCreateProgramWithBinary_Enter ();
		r = real_clCreateProgramWithBinary (c, n, dl, l, b, bs, e);
		Extrae_Probe_clCreateProgramWithBinary_Exit ();
	}
	else if (!mpitrace_on && real_clCreateProgramWithBinary != NULL)
	{
		r = real_clCreateProgramWithBinary (c, n, dl, l, b, bs, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateProgramWithBinary was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_program clCreateProgramWithBuiltInKernels (cl_context c, cl_uint n,
	const cl_device_id *dl, const char *kn, cl_int *e)
{
	cl_program r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateProgramWithBuiltInKernels (real at %p)\n", real_clCreateProgramWithBuiltInKernels);
#endif

	if (mpitrace_on && real_clCreateProgramWithBuiltInKernels != NULL)
	{
		Extrae_Probe_clCreateProgramWithBuiltInKernels_Enter ();
		r = real_clCreateProgramWithBuiltInKernels (c, n, dl, kn, e);
		Extrae_Probe_clCreateProgramWithBuiltInKernels_Exit ();
	}
	else if (!mpitrace_on && real_clCreateProgramWithBuiltInKernels != NULL)
	{
		r = real_clCreateProgramWithBuiltInKernels (c, n, dl, kn, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCreateProgramWithBuilInKernels was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueFillBuffer (cl_command_queue c, cl_mem m, const void *ptr, 
	size_t ps, size_t o, size_t s, cl_uint n, const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueFillBuffer (real at %p)\n", real_clEnqueueFillBuffer);
#endif

	if (mpitrace_on && real_clEnqueueFillBuffer != NULL)
	{
		Extrae_Probe_clEnqueueFillBuffer_Enter ();
		r = real_clEnqueueFillBuffer (c, m, ptr, ps, o, s, n, ewl, e);
		Extrae_Probe_clEnqueueFillBuffer_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueFillBuffer != NULL)
	{
		r = real_clEnqueueFillBuffer (c, m, ptr, ps, o, s, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueFillBuffer was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueCopyBuffer (cl_command_queue c, cl_mem src, cl_mem dst, 
	size_t so, size_t dso, size_t s, cl_uint n, const cl_event *e, cl_event *evt)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueCopyBuffer (real at %p)\n", real_clEnqueueCopyBuffer);
#endif

	if (mpitrace_on && real_clEnqueueCopyBuffer != NULL)
	{
		Extrae_Probe_clEnqueueCopyBuffer_Enter ();
		r = real_clEnqueueCopyBuffer (c, src, dst, so, dso, s, n, e, evt);
		Extrae_Probe_clEnqueueCopyBuffer_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueCopyBuffer != NULL)
	{
		r = real_clEnqueueCopyBuffer (c, src, dst, so, dso, s, n, e, evt);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueCopyBuffer was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueCopyBufferRect (cl_command_queue c, cl_mem src, cl_mem dst,
	const size_t *s, const size_t *d, const size_t *r, size_t srp, size_t ssp,
	size_t drp, size_t dsp, cl_uint n, const cl_event *ewl, cl_event *e)
{
	cl_int res;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug clEnqueueCopyBufferRect (real at %p)\n", real_clEnqueueCopyBufferRect);
#endif

	if (mpitrace_on && real_clEnqueueCopyBufferRect != NULL)
	{
		Extrae_Probe_clEnqueueCopyBufferRect_Enter ();
		res = real_clEnqueueCopyBufferRect (c, src, dst, s, d, r, srp, ssp,
		  drp, dsp, n, ewl, e);
		Extrae_Probe_clEnqueueCopyBufferRect_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueCopyBufferRect != NULL)
	{
		res = real_clEnqueueCopyBufferRect (c, src, dst, s, d, r, srp, ssp,
		  drp, dsp, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueCopyBufferRect was not hooked!\n");
		exit (-1);
	}

	return res;
}

cl_int clEnqueueNDRangeKernel (cl_command_queue c, cl_kernel k, cl_uint n,
	const size_t *gwo, const size_t *gws, const size_t *lws, cl_uint ne,
	const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueNDRangeKernel (real at %p)\n", real_clEnqueueNDRangeKernel);
#endif

	if (mpitrace_on && real_clEnqueueNDRangeKernel != NULL)
	{
		Extrae_Probe_clEnqueueNDRangeKernel_Enter ();
		r = real_clEnqueueNDRangeKernel (c, k, n, gwo, gws, lws, ne, ewl, e);
		Extrae_Probe_clEnqueueNDRangeKernel_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueNDRangeKernel != NULL)
	{
		r = real_clEnqueueNDRangeKernel (c, k, n, gwo, gws, lws, ne, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueNDRangeKernel was not hooked!\n");
		exit (-1);
	}
	return r;
}

cl_int clEnqueueTask (cl_command_queue c, cl_kernel k, cl_uint n, 
	const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueTask (real at %p)\n", real_clEnqueueTask);
#endif

	if (mpitrace_on && real_clEnqueueTask != NULL)
	{
		Extrae_Probe_clEnqueueTask_Enter ();
		r = real_clEnqueueTask (c, k, n, ewl, e);
		Extrae_Probe_clEnqueueTask_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueTask != NULL)
	{
		r = real_clEnqueueTask (c, k, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueTask was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueNativeKernel (cl_command_queue c,
	void (*ptr)(void *),
	void *args, size_t cb, cl_uint nmo, const cl_mem *ml, const void **aml,
	cl_uint newl, const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEqneueNativeKernel (real at %p)\n", real_clEnqueueNativeKernel);
#endif

	if (mpitrace_on && real_clEnqueueNativeKernel != NULL)
	{
		Extrae_Probe_clEnqueueNativeKernel_Enter ();
		r = real_clEnqueueNativeKernel (c, ptr, args, cb, nmo, ml, aml, newl, ewl, e);
		Extrae_Probe_clEnqueueNativeKernel_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueNativeKernel != NULL)
	{
		r = real_clEnqueueNativeKernel (c, ptr, args, cb, nmo, ml, aml, newl, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueNativeKernel was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueReadBuffer (cl_command_queue c, cl_mem m, cl_bool b, size_t o,
	size_t s, void *p, cl_uint u, const cl_event *e, cl_event *evt)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueReadBuffer (real at %p)\n", real_clEnqueueReadBuffer);
#endif

	if (mpitrace_on && real_clEnqueueReadBuffer != NULL)
	{
		Extrae_Probe_clEnqueueReadBuffer_Enter ();
		r = real_clEnqueueReadBuffer (c, m, b, o, s, p, u, e, evt);
		Extrae_Probe_clEnqueueReadBuffer_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueReadBuffer != NULL)
	{
		r = real_clEnqueueReadBuffer (c, m, b, o, s, p, u, e, evt);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueReadBuffer was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueReadBufferRect (cl_command_queue c, cl_mem m, cl_bool b,
	const size_t *bo, const size_t *ho, const size_t *r, size_t brp,
	size_t bsp, size_t hrp, size_t hsp, void *ptr, cl_uint n, 
	const cl_event *ewl, cl_event *e)
{
	cl_int res;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueReadBufferRect (real at %p)\n", real_clEnqueueReadBufferRect);
#endif

	if (mpitrace_on && real_clEnqueueReadBufferRect != NULL)
	{
		Extrae_Probe_clEnqueueReadBufferRect_Enter ();
		res = real_clEnqueueReadBufferRect (c, m, b, bo, ho, r, brp, bsp, hrp,
		  hsp, ptr, n, ewl ,e);
		Extrae_Probe_clEnqueueReadBufferRect_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueReadBufferRect != NULL)
	{
		res = real_clEnqueueReadBufferRect (c, m, b, bo, ho, r, brp, bsp, hrp,
		  hsp, ptr, n, ewl ,e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueReadBufferRect was not hooked!\n");
		exit (-1);
	}

	return res;
}

cl_int clEnqueueWriteBuffer (cl_command_queue c, cl_mem m, cl_bool b, size_t o,
	size_t s, const void *p, cl_uint u, const cl_event *e, cl_event *evt)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueWriteBuffer (real at %p)\n", real_clEnqueueWriteBuffer);
#endif

	if (mpitrace_on && real_clEnqueueWriteBuffer != NULL)
	{
		Extrae_Probe_clEnqueueWriteBuffer_Enter ();
		r = real_clEnqueueWriteBuffer (c, m, b, o, s, p, u, e, evt);
		Extrae_Probe_clEnqueueWriteBuffer_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueWriteBuffer != NULL)
	{
		r = real_clEnqueueWriteBuffer (c, m, b, o, s, p, u, e, evt);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueWriteBuffer was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueWriteBufferRect (cl_command_queue c, cl_mem m, cl_bool b,
	const size_t *bo, const size_t *ho, const size_t *r, size_t brp,
	size_t bsp, size_t hrp, size_t hsp, const void *ptr, cl_uint n, 
	const cl_event *ewl, cl_event *e)
{
	cl_int res;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueWriteBufferRect (real at %p)\n", real_clEnqueueWriteBufferRect);
#endif

	if (mpitrace_on && real_clEnqueueWriteBufferRect != NULL)
	{
		Extrae_Probe_clEnqueueWriteBufferRect_Enter ();
		res = real_clEnqueueWriteBufferRect (c, m, b, bo, ho, r, brp, bsp,
		  hrp, hsp, ptr, n, ewl, e);
		Extrae_Probe_clEnqueueWriteBufferRect_Exit ();
	}
	else if (!mpitrace_on && real_clEnqueueWriteBufferRect != NULL)
	{
		res = real_clEnqueueWriteBufferRect (c, m, b, bo, ho, r, brp, bsp,
		  hrp, hsp, ptr, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueWriteBufferRect was not hooked!\n");
		exit (-1);
	}

	return res;
}

cl_int clBuildProgram (cl_program p, cl_uint n, const cl_device_id *dl,
	const char *o,
	void (*cbk)(cl_program, void *), 
	void *ud)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clBuildProgram (real at %p)\n", real_clBuildProgram);
#endif

	if (mpitrace_on && real_clBuildProgram != NULL)
	{
		Extrae_Probe_clBuildProgram_Enter ();
		r = real_clBuildProgram (p, n, dl, o, cbk, ud);
		Extrae_Probe_clBuildProgram_Exit ();
	}
	else if (!mpitrace_on && real_clBuildProgram != NULL)
	{
		r = real_clBuildProgram (p, n, dl, o, cbk, ud);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clBuildProgram was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clCompileProgram (cl_program p, cl_uint n, const cl_device_id *dl,
	const char *o, cl_uint nih, const cl_program *ih, const char **hin,
	void (*cbk)(cl_program, void *), 
	void *ud)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCompileProgram (real at %p)\n", real_clCompileProgram);
#endif

	if (mpitrace_on && real_clCompileProgram != NULL)
	{
		Extrae_Probe_clCompileProgram_Enter ();
		r = real_clCompileProgram (p, n, dl, o, nih, ih, hin, cbk, ud);
		Extrae_Probe_clCompileProgram_Exit ();
	}
	else if (!mpitrace_on && real_clCompileProgram != NULL)
	{
		r = real_clCompileProgram (p, n, dl, o, nih, ih, hin, cbk, ud);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clCompileProgram was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_program clLinkProgram (cl_context c, cl_uint n, const cl_device_id *dl,
	const char *o, cl_uint nip, const cl_program *ip,
	void (*cbk)(cl_program, void *), 
	void *ud, cl_int *e)
{
	cl_program r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clLinkProgram (real at %p)\n", real_clLinkProgram);
#endif

	if (mpitrace_on && real_clLinkProgram != NULL)
	{
		Extrae_Probe_clLinkProgram_Enter ();
		r = real_clLinkProgram (c, n, dl, o, nip, ip, cbk, ud, e);
		Extrae_Probe_clLinkProgram_Exit ();
	}
	else if (!mpitrace_on && real_clLinkProgram != NULL)
	{
		r = real_clLinkProgram (c, n, dl, o, nip, ip, cbk, ud, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clLinkProgram was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clFinish (cl_command_queue q)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clFinish (real at %p)\n", real_clFinish);
#endif

	if (mpitrace_on && real_clFinish != NULL)
	{
		Extrae_Probe_clFinish_Enter ();
		r = real_clFinish (q);
		Extrae_Probe_clFinish_Exit ();
	}
	else if (!mpitrace_on && real_clFinish != NULL)
	{
		r = real_clFinish (q);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clFinish was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clFlush (cl_command_queue q)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clFlush (real at %p)\n", real_clFlush);
#endif

	if (mpitrace_on && real_clFlush != NULL)
	{
		Extrae_Probe_clFlush_Enter ();
		r = real_clFlush (q);
		Extrae_Probe_clFlush_Exit ();
	}
	else if (!mpitrace_on && real_clFlush != NULL)
	{
		r = real_clFlush (q);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clFlush was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clWaitForEvents (cl_uint n, const cl_event *el)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug: clWaitForEvents (real at %p)\n", real_clWaitForEvents);
#endif

	if (mpitrace_on && real_clWaitForEvents != NULL)
	{
		Extrae_Probe_clWaitForEvents_Enter ();
		r = real_clWaitForEvents (n, el);
		Extrae_Probe_clWaitForEvents_Exit ();
	}
	else if (!mpitrace_on && real_clWaitForEvents != NULL)
	{
		r = real_clWaitForEvents (n, el);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clWaitForEvents was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueMarkerWithWaitList (cl_command_queue q, cl_uint n,
	const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug: clEnqueueMarkerWithWaitList (real at %p)\n", real_clEnqueueMarkerWithWaitList);
#endif

	if (mpitrace_on && real_clEnqueueMarkerWithWaitList != NULL)
	{
		Extrae_Probe_clEnqueueMarkerWithWaitList_Enter ();
		r = real_clEnqueueMarkerWithWaitList (q, n, ewl, e);
		Extrae_Probe_clEnqueueMarkerWithWaitList_Enter ();
	}
	else if (!mpitrace_on && real_clEnqueueMarkerWithWaitList != NULL)
	{
		r = real_clEnqueueMarkerWithWaitList (q, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueMarkerWithWaitList!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueBarrierWithWaitList (cl_command_queue q, cl_uint n,
	const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug: clEnqueueBarrierWithWaitList (real at %p)\n", real_clEnqueueBarrierWithWaitList);
#endif

	if (mpitrace_on && real_clEnqueueMarkerWithWaitList != NULL)
	{
		Extrae_Probe_clEnqueueBarrierWithWaitList_Enter ();
		r = real_clEnqueueBarrierWithWaitList (q, n, ewl, e);
		Extrae_Probe_clEnqueueBarrierWithWaitList_Enter ();
	}
	else if (!mpitrace_on && real_clEnqueueMarkerWithWaitList != NULL)
	{
		r = real_clEnqueueBarrierWithWaitList (q, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueBarrierWithWaitList!\n");
		exit (-1);
	}

	return r;
}

