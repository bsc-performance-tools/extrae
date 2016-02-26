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

#include "wrapper.h"
#include "opencl_probe.h"
#include "opencl_common.h"
#include "opencl_wrapper.h"

#if defined(PIC)
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
#ifdef CL_VERSION_1_2
static cl_int (*real_clEnqueueMarkerWithWaitList)(cl_command_queue, cl_uint, const cl_event *, cl_event *) = NULL;
static cl_int (*real_clEnqueueBarrierWithWaitList)(cl_command_queue, cl_uint, const cl_event *, cl_event *) = NULL;
#endif
static cl_int (*real_clEnqueueMarker)(cl_command_queue, cl_event *) = NULL;
static cl_int (*real_clEnqueueBarrier)(cl_command_queue) = NULL;
static void* (*real_clEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *) = NULL;
static cl_int (*real_clEnqueueUnmapMemObject)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *) = NULL;
#ifdef CL_VERSION_1_2
static cl_int (*real_clEnqueueMigrateMemObjects)(cl_command_queue, cl_uint, const cl_mem *, cl_mem_migration_flags, cl_uint, const cl_event *, cl_event *) = NULL;
#endif
static cl_int (*real_clRetainCommandQueue)(cl_command_queue) = NULL;
static cl_int (*real_clReleaseCommandQueue)(cl_command_queue) = NULL;
static cl_int (*real_clRetainContext)(cl_context) = NULL;
static cl_int (*real_clReleaseContext)(cl_context) = NULL;
static cl_int (*real_clRetainDevice)(cl_device_id) = NULL;
static cl_int (*real_clReleaseDevice)(cl_device_id) = NULL;
static cl_int (*real_clRetainEvent)(cl_event) = NULL;
static cl_int (*real_clReleaseEvent)(cl_event) = NULL;
static cl_int (*real_clRetainKernel)(cl_kernel) = NULL;
static cl_int (*real_clReleaseKernel)(cl_kernel) = NULL;
static cl_int (*real_clRetainMemObject)(cl_mem) = NULL;
static cl_int (*real_clReleaseMemObject)(cl_mem) = NULL;
static cl_int (*real_clRetainProgram)(cl_program) = NULL;
static cl_int (*real_clReleaseProgram)(cl_program) = NULL;

static int Extrae_Prepare_CommandQueue = FALSE;

#endif /* PIC */

void Extrae_OpenCL_fini (void)
{
	Extrae_OpenCL_clQueueFlush_All();
}

void Extrae_OpenCL_init (unsigned rank)
{
	UNREFERENCED_PARAMETER(rank);

#if defined(PIC)

#if defined(__APPLE__)
	void *lib = dlopen("/System/Libraries/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW);
#else
	void *lib = RTLD_NEXT;
#endif

	real_clCreateBuffer = (cl_mem(*)(cl_context, cl_mem_flags, size_t, void*, cl_int *))
		dlsym (lib, "clCreateBuffer");

	real_clCreateCommandQueue = (cl_command_queue(*)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*))
		dlsym (lib, "clCreateCommandQueue");

	real_clCreateContext = (cl_context(*)(const cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *))
		dlsym (lib, "clCreateContext");

	real_clCreateContextFromType = (cl_context(*)(const cl_context_properties *, cl_device_type, void *, void *, cl_int *))
		dlsym (lib, "clCreateContextFromType");

	real_clCreateKernel = (cl_kernel(*)(cl_program, const char *, cl_int *))
		dlsym (lib, "clCreateKernel");

	real_clCreateKernelsInProgram = (cl_int(*)(cl_program, cl_uint, cl_kernel *, cl_uint *))
		dlsym (lib, "clCreateKernelsInProgram");

	real_clSetKernelArg = (cl_int(*)(cl_kernel, cl_uint, size_t, const void *))
		dlsym (lib, "clSetKernelArg");

	real_clCreateProgramWithSource = (cl_program(*)(cl_context, cl_uint, const char **, const size_t *, cl_int *))
		dlsym (lib, "clCreateProgramWithSource");

	real_clCreateProgramWithBinary = (cl_program(*)(cl_context, cl_uint, const cl_device_id *, const size_t *, const unsigned char **, cl_int *, cl_int *))
		dlsym (lib, "clCreateProgramWithBinary");

	real_clCreateProgramWithBuiltInKernels = (cl_program(*)(cl_context, cl_uint, const cl_device_id *, const char *, cl_int *))
		dlsym (lib, "clCreateProgramWithBuiltInKernels");

	real_clCreateSubBuffer = (cl_mem(*)(cl_mem, cl_mem_flags, cl_buffer_create_type, const void *, cl_int *))
		dlsym (lib, "clCreateSubBuffer");

	real_clEnqueueFillBuffer = (cl_int(*)(cl_command_queue, cl_mem, const void *, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueFillBuffer");

	real_clEnqueueCopyBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueCopyBuffer");

	real_clEnqueueCopyBufferRect = (cl_int(*)(cl_command_queue, cl_mem, cl_mem, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueCopyBufferRect");

	real_clEnqueueNDRangeKernel = (cl_int(*)(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueNDRangeKernel");

	real_clEnqueueTask = (cl_int(*)(cl_command_queue, cl_kernel, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueTask");

	real_clEnqueueNativeKernel = (cl_int(*)(cl_command_queue, void *, void *, size_t, cl_uint, const cl_mem *, const void **, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueNativeKernel");

	real_clEnqueueReadBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueReadBuffer");

	real_clEnqueueReadBufferRect = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueReadBufferRect");

	real_clEnqueueWriteBuffer = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueWriteBuffer");

	real_clEnqueueWriteBufferRect = (cl_int(*)(cl_command_queue, cl_mem, cl_bool, const size_t *, const size_t *, const size_t *, size_t, size_t, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueWriteBufferRect");

	real_clBuildProgram = (cl_int(*)(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *))
		dlsym (lib, "clBuildProgram");

	real_clCompileProgram = (cl_int(*)(cl_program, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *, const char **, void *, void *))
		dlsym (lib, "clCompileProgram");

	real_clLinkProgram = (cl_program(*)(cl_context, cl_uint, const cl_device_id *, const char *, cl_uint, const cl_program *, void *, void *, cl_int *))
		dlsym (lib, "clLinkProgram");

	real_clFinish = (cl_int(*)(cl_command_queue))
		dlsym (lib, "clFinish");

	real_clFlush = (cl_int(*)(cl_command_queue))
		dlsym (lib, "clFlush");

	real_clWaitForEvents = (cl_int(*)(cl_uint, const cl_event *el))
		dlsym (lib, "clWaitForEvents");

#ifdef CL_VERSION_1_2
	real_clEnqueueMarkerWithWaitList = (cl_int(*)(cl_command_queue, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueMarkerWithWaitList");

	real_clEnqueueBarrierWithWaitList = (cl_int(*)(cl_command_queue, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueBarrierWithWaitList");
#endif

	real_clEnqueueMarker = (cl_int(*)(cl_command_queue, cl_event *))
		dlsym (lib, "clEnqueueMarker");

	real_clEnqueueBarrier = (cl_int(*)(cl_command_queue))
		dlsym (lib, "clEnqueueBarrier");

	real_clEnqueueMapBuffer = (void* (*)(cl_command_queue, cl_mem, cl_bool, cl_map_flags, size_t, size_t, cl_uint, const cl_event *, cl_event *, cl_int *))
		dlsym (lib, "clEnqueueMapBuffer");

	real_clEnqueueUnmapMemObject = (cl_int (*)(cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueUnmapMemObject");

#ifdef CL_VERSION_1_2
	real_clEnqueueMigrateMemObjects = (cl_int (*)(cl_command_queue, cl_uint, const cl_mem *, cl_mem_migration_flags, cl_uint, const cl_event *, cl_event *))
		dlsym (lib, "clEnqueueMigrateMemObjects");
#endif

	real_clRetainCommandQueue = (cl_int(*)(cl_command_queue))
	  dlsym (lib, "clRetainCommandQueue");

	real_clReleaseCommandQueue = (cl_int(*)(cl_command_queue))
	  dlsym (lib, "clReleaseCommandQueue");

	real_clRetainContext = (cl_int(*)(cl_context))
	  dlsym (lib, "clRetainContext");

	real_clReleaseContext = (cl_int(*)(cl_context))
	  dlsym (lib, "clReleaseContext");

	real_clRetainDevice = (cl_int(*)(cl_device_id))
	  dlsym (lib, "clRetainDevice");

	real_clReleaseDevice = (cl_int(*)(cl_device_id))
	  dlsym (lib, "clReleaseDevice");

	real_clRetainEvent = (cl_int(*)(cl_event))
	  dlsym (lib, "clRetainEvent");

	real_clReleaseEvent = (cl_int(*)(cl_event))
	  dlsym (lib, "clReleaseEvent");

	real_clRetainKernel = (cl_int(*)(cl_kernel))
	  dlsym (lib, "clRetainKernel");

	real_clReleaseKernel = (cl_int(*)(cl_kernel))
	  dlsym (lib, "clReleaseKernel");

	real_clRetainMemObject = (cl_int(*)(cl_mem))
	  dlsym (lib, "clRetainMemObject");

	real_clReleaseMemObject = (cl_int(*)(cl_mem))
	  dlsym (lib, "clReleaseMemObject");

	real_clRetainProgram = (cl_int(*)(cl_program))
	  dlsym (lib, "clRetainProgram");

	real_clReleaseProgram = (cl_int(*)(cl_program))
	  dlsym (lib, "clReleaseProgram");
#else
	fprintf (stderr, PACKAGE_NAME": Warning! OpenCL instrumentation requires linking with shared library!\n");
#endif /* PIC */
}

/*
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
	INJECTED CODE -- INJECTED CODE -- INJECTED CODE -- INJECTED CODE
*/

#if defined(PIC)

cl_mem clCreateBuffer (cl_context c, cl_mem_flags m, size_t s, void *p, 
	cl_int *e)
{
	cl_mem r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateBuffer (real at %p)\n", real_clCreateBuffer);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateBuffer != NULL)
	{
		Extrae_Probe_clCreateBuffer_Enter();
		r = real_clCreateBuffer (c, m, s, p, e);
		Extrae_Probe_clCreateBuffer_Exit();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateBuffer != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateCommandQueue != NULL)
	{
		Extrae_Probe_clCreateCommandQueue_Enter ();
		Extrae_Prepare_CommandQueue = TRUE;
		r = real_clCreateCommandQueue (c, d, p, e);
		Extrae_OpenCL_clCreateCommandQueue (r, d, p);
		Extrae_Prepare_CommandQueue = FALSE;
		Extrae_Probe_clCreateCommandQueue_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateCommandQueue != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateContext != NULL)
	{
		Extrae_Probe_clCreateContext_Enter();
		r = real_clCreateContext (p, n, d, pfn, udata, e);
		Extrae_Probe_clCreateContext_Exit();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateContext != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateContextFromType != NULL)
	{
		Extrae_Probe_clCreateContextFromType_Enter ();
		r = real_clCreateContextFromType (p, dt, pfn, udata, e);
		Extrae_Probe_clCreateContextFromType_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateContextFromType != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateSubBuffer != NULL)
	{
		Extrae_Probe_clCreateSubBuffer_Enter ();
		r = real_clCreateSubBuffer (m, mf, bct, b, e);
		Extrae_Probe_clCreateSubBuffer_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateSubBuffer != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateKernel != NULL)
	{
		Extrae_Probe_clCreateKernel_Enter ();
		r = real_clCreateKernel (p, k, e);
		Extrae_Probe_clCreateKernel_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateKernel != NULL)
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

cl_int clCreateKernelsInProgram (cl_program p, cl_uint n, cl_kernel *ks, cl_uint *nks)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clCreateKernelsInProgram (%p)\n", real_clCreateKernelsInProgram);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateKernelsInProgram != NULL)
	{
		Extrae_Probe_clCreateKernelsInProgram_Enter ();
		r = real_clCreateKernelsInProgram (p, n, ks, nks);
		Extrae_Probe_clCreateKernelsInProgram_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateKernelsInProgram != NULL)
	{
		r = real_clCreateKernelsInProgram (p, n, ks, nks);
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clSetKernelArg != NULL)
	{
		Extrae_Probe_clSetKernelArg_Enter ();
		r = real_clSetKernelArg (k, a, as, av);
		Extrae_Probe_clSetKernelArg_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clSetKernelArg != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateProgramWithSource != NULL)
	{
		Extrae_Probe_clCreateProgramWithSource_Enter ();
		r = real_clCreateProgramWithSource (c, u, s, l, e);
		Extrae_Probe_clCreateProgramWithSource_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateProgramWithSource != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateProgramWithBinary != NULL)
	{
		Extrae_Probe_clCreateProgramWithBinary_Enter ();
		r = real_clCreateProgramWithBinary (c, n, dl, l, b, bs, e);
		Extrae_Probe_clCreateProgramWithBinary_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateProgramWithBinary != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCreateProgramWithBuiltInKernels != NULL)
	{
		Extrae_Probe_clCreateProgramWithBuiltInKernels_Enter ();
		r = real_clCreateProgramWithBuiltInKernels (c, n, dl, kn, e);
		Extrae_Probe_clCreateProgramWithBuiltInKernels_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCreateProgramWithBuiltInKernels != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueFillBuffer != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueFillBuffer_Enter ();
		r = real_clEnqueueFillBuffer (c, m, ptr, ps, o, s, n, ewl, &evt);
		Extrae_OpenCL_addEventToQueue (c, evt, OPENCL_CLENQUEUEFILLBUFFER_ACC_EV);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueFillBuffer_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueFillBuffer != NULL)
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
	size_t so, size_t dso, size_t s, cl_uint n, const cl_event *e, cl_event *ev)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueCopyBuffer (real at %p)\n", real_clEnqueueCopyBuffer);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueCopyBuffer != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueCopyBuffer_Enter ();
		r = real_clEnqueueCopyBuffer (c, src, dst, so, dso, s, n, e, &evt);
		Extrae_OpenCL_addEventToQueue (c, evt, OPENCL_CLENQUEUECOPYBUFFER_ACC_EV);
		if (ev != NULL)
			*ev = evt;
		Extrae_Probe_clEnqueueCopyBuffer_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueCopyBuffer != NULL)
	{
		r = real_clEnqueueCopyBuffer (c, src, dst, so, dso, s, n, e, ev);
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueCopyBufferRect != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueCopyBufferRect_Enter ();
		res = real_clEnqueueCopyBufferRect (c, src, dst, s, d, r, srp, ssp,
		  drp, dsp, n, ewl, &evt);
		Extrae_OpenCL_addEventToQueue (c, evt, OPENCL_CLENQUEUECOPYBUFFERRECT_ACC_EV);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueCopyBufferRect_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueCopyBufferRect != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueNDRangeKernel != NULL)
	{
		cl_event evt;
		unsigned kid = 0;
		Extrae_OpenCL_annotateKernelName (k, &kid);
		kid++;
		
		Extrae_Probe_clEnqueueNDRangeKernel_Enter (kid);

		TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV,
		  TASKID, 0, Extrae_OpenCL_tag_generator(),
		  Extrae_OpenCL_tag_generator());

		r = real_clEnqueueNDRangeKernel (c, k, n, gwo, gws, lws, ne, ewl, &evt);
		Extrae_OpenCL_addEventToQueueWithKernel (c, evt, OPENCL_CLENQUEUENDRANGEKERNEL_ACC_EV, k);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueNDRangeKernel_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueNDRangeKernel != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueTask != NULL)
	{
		cl_event evt;
		unsigned kid = 0;
		Extrae_OpenCL_annotateKernelName (k, &kid);
		kid++;

		Extrae_Probe_clEnqueueTask_Enter (kid);

		TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV,
		  TASKID, 0, Extrae_OpenCL_tag_generator(),
		  Extrae_OpenCL_tag_generator());

		r = real_clEnqueueTask (c, k, n, ewl, &evt);
		Extrae_OpenCL_addEventToQueueWithKernel (c, evt, OPENCL_CLENQUEUETASK_ACC_EV, k);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueTask_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueTask != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueNativeKernel != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueNativeKernel_Enter ();

		TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV,
		  TASKID, 0, Extrae_OpenCL_tag_generator(),
		  Extrae_OpenCL_tag_generator());

		r = real_clEnqueueNativeKernel (c, ptr, args, cb, nmo, ml, aml, newl, ewl, &evt);
		Extrae_OpenCL_addEventToQueue (c, evt, OPENCL_CLENQUEUENATIVEKERNEL_ACC_EV);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueNativeKernel_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueNativeKernel != NULL)
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
	size_t s, void *p, cl_uint u, const cl_event *e, cl_event *ev)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueReadBuffer (real at %p)\n", real_clEnqueueReadBuffer);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueReadBuffer != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueReadBuffer_Enter (b, s);
		r = real_clEnqueueReadBuffer (c, m, b, o, s, p, u, e, &evt);

		Extrae_OpenCL_addEventToQueueWithSize (c, evt,
		  b?OPENCL_CLENQUEUEREADBUFFER_ACC_EV:OPENCL_CLENQUEUEREADBUFFER_ASYNC_ACC_EV,
		  s);

		if (ev != NULL)
			*ev = evt;
		if (b && !Extrae_OpenCL_Queue_OoO (c))
			Extrae_OpenCL_clQueueFlush (c, FALSE);

		Extrae_Probe_clEnqueueReadBuffer_Exit (b);
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME,
			  USER_RECV_EV, TASKID, s,
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueReadBuffer != NULL)
	{
		r = real_clEnqueueReadBuffer (c, m, b, o, s, p, u, e, ev);
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueReadBufferRect != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueReadBufferRect_Enter (b);
		res = real_clEnqueueReadBufferRect (c, m, b, bo, ho, r, brp, bsp, hrp,
		  hsp, ptr, n, ewl, &evt);
		Extrae_OpenCL_addEventToQueue (c, evt,
		  b?OPENCL_CLENQUEUEREADBUFFERRECT_ACC_EV:OPENCL_CLENQUEUEREADBUFFERRECT_ASYNC_ACC_EV);

		if (e != NULL)
			*e = evt;
		if (b && !Extrae_OpenCL_Queue_OoO (c))
			Extrae_OpenCL_clQueueFlush (c, FALSE);

		Extrae_Probe_clEnqueueReadBufferRect_Exit (b);
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME,
			  USER_RECV_EV, TASKID, 0,
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueReadBufferRect != NULL)
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
	size_t s, const void *p, cl_uint u, const cl_event *e, cl_event *ev)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueWriteBuffer (real at %p)\n", real_clEnqueueWriteBuffer);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueWriteBuffer != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueWriteBuffer_Enter (b, s);
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME,
			  USER_SEND_EV, TASKID, s,
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());

		r = real_clEnqueueWriteBuffer (c, m, b, o, s, p, u, e, &evt);

		Extrae_OpenCL_addEventToQueueWithSize (c, evt,
		  b?OPENCL_CLENQUEUEWRITEBUFFER_ACC_EV:OPENCL_CLENQUEUEWRITEBUFFER_ASYNC_ACC_EV,
		  s);
		if (ev != NULL)
			*ev = evt;
/*
		This is not sure, when writebuffer returns it means that it is sent to
		the accel, but it does not need to be finished.

		if (b && !Extrae_OpenCL_Queue_OoO (c))
			Extrae_OpenCL_clQueueFlush (c);
*/
		Extrae_Probe_clEnqueueWriteBuffer_Exit (b);
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueWriteBuffer != NULL)
	{
		r = real_clEnqueueWriteBuffer (c, m, b, o, s, p, u, e, ev);
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueWriteBufferRect != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueWriteBufferRect_Enter (b);
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME,
			  USER_SEND_EV, TASKID, 0,
			  Extrae_OpenCL_tag_generator(),
			  Extrae_OpenCL_tag_generator());

		res = real_clEnqueueWriteBufferRect (c, m, b, bo, ho, r, brp, bsp,
		  hrp, hsp, ptr, n, ewl, &evt);

		Extrae_OpenCL_addEventToQueue (c, evt,
		  b?OPENCL_CLENQUEUEWRITEBUFFERRECT_ACC_EV:OPENCL_CLENQUEUEWRITEBUFFERRECT_ASYNC_ACC_EV);
		if (e != NULL)
			*e = evt;
/*
		This is not sure, when writebuffer returns it means that it is sent to
		the accel, but it does not need to be finished.

		if (b && !Extrae_OpenCL_Queue_OoO (c))
			Extrae_OpenCL_clQueueFlush (c);
*/
		Extrae_Probe_clEnqueueWriteBufferRect_Exit (b);
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueWriteBufferRect != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clBuildProgram != NULL)
	{
		Extrae_Probe_clBuildProgram_Enter ();
		r = real_clBuildProgram (p, n, dl, o, cbk, ud);
		Extrae_Probe_clBuildProgram_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clBuildProgram != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clCompileProgram != NULL)
	{
		Extrae_Probe_clCompileProgram_Enter ();
		r = real_clCompileProgram (p, n, dl, o, nih, ih, hin, cbk, ud);
		Extrae_Probe_clCompileProgram_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clCompileProgram != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clLinkProgram != NULL)
	{
		Extrae_Probe_clLinkProgram_Enter ();
		r = real_clLinkProgram (c, n, dl, o, nip, ip, cbk, ud, e);
		Extrae_Probe_clLinkProgram_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clLinkProgram != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clFinish != NULL)
	{
		if (!Extrae_Prepare_CommandQueue)
			Extrae_Probe_clFinish_Enter (
			  Extrae_OpenCL_lookForOpenCLQueueToThreadID (q));

		r = real_clFinish (q);

		if (!Extrae_Prepare_CommandQueue)
		{
			Extrae_Probe_clFinish_Exit ();
			Extrae_OpenCL_clQueueFlush (q, TRUE);
		}
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clFinish != NULL)
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

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clFlush != NULL)
	{
		Extrae_Probe_clFlush_Enter ();
		r = real_clFlush (q);
		Extrae_Probe_clFlush_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clFlush != NULL)
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
	fprintf (stderr, PACKAGE_NAME": Debug : clWaitForEvents (real at %p)\n", real_clWaitForEvents);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clWaitForEvents != NULL)
	{
		Extrae_Probe_clWaitForEvents_Enter ();
		r = real_clWaitForEvents (n, el);
		Extrae_Probe_clWaitForEvents_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clWaitForEvents != NULL)
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

#ifdef CL_VERSION_1_2
cl_int clEnqueueMarkerWithWaitList (cl_command_queue q, cl_uint n,
	const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueMarkerWithWaitList (real at %p)\n", real_clEnqueueMarkerWithWaitList);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueMarkerWithWaitList != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueMarkerWithWaitList_Enter ();
		r = real_clEnqueueMarkerWithWaitList (q, n, ewl, &evt);
		Extrae_OpenCL_addEventToQueue (q, evt, OPENCL_CLENQUEUEMARKERWITHWAITLIST_ACC_EV);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueMarkerWithWaitList_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueMarkerWithWaitList != NULL)
	{
		r = real_clEnqueueMarkerWithWaitList (q, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueMarkerWithWaitList was not hooked!\n");
		exit (-1);
	}

	return r;
}
#endif

#ifdef CL_VERSION_1_2
cl_int clEnqueueBarrierWithWaitList (cl_command_queue q, cl_uint n,
	const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueBarrierWithWaitList (real at %p)\n", real_clEnqueueBarrierWithWaitList);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueBarrierWithWaitList != NULL)
	{
		cl_event evt;

		if (!Extrae_Prepare_CommandQueue)
			Extrae_Probe_clEnqueueBarrierWithWaitList_Enter ();
		r = real_clEnqueueBarrierWithWaitList (q, n, ewl, &evt);
		if (!Extrae_Prepare_CommandQueue)
		{
			Extrae_OpenCL_addEventToQueue (q, evt, OPENCL_CLENQUEUEBARRIERWITHWAITLIST_ACC_EV);
			if (e != NULL)
				*e = evt;
			Extrae_Probe_clEnqueueBarrierWithWaitList_Exit ();
		}
		else
		{
			if (e != NULL)
				*e = evt;
		}
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueBarrierWithWaitList != NULL)
	{
		r = real_clEnqueueBarrierWithWaitList (q, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueBarrierWithWaitList was not hooked!\n");
		exit (-1);
	}

	return r;
}
#endif

cl_int clEnqueueMarker (cl_command_queue q, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueMarker (real at %p)\n", real_clEnqueueMarker);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueMarker != NULL)
	{
		cl_event evt;

		if (!Extrae_Prepare_CommandQueue)
			Extrae_Probe_clEnqueueMarker_Enter ();
		r = real_clEnqueueMarker (q, &evt);
		if (!Extrae_Prepare_CommandQueue)
		{
			Extrae_Probe_clEnqueueMarker_Exit ();
			Extrae_OpenCL_addEventToQueue (q, evt, OPENCL_CLENQUEUEMARKER_ACC_EV);
			if (e != NULL)
				*e = evt;
		}
		else
		{
			if (e != NULL)
				*e = evt;
		}
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueMarker != NULL)
	{
		r = real_clEnqueueMarker (q, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueMarker was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueBarrier (cl_command_queue q)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueBarrier (real at %p)\n", real_clEnqueueBarrier);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueBarrier != NULL)
	{
		if (!Extrae_Prepare_CommandQueue)
			Extrae_Probe_clEnqueueBarrier_Enter ();
		r = real_clEnqueueBarrier (q);
		if (!Extrae_Prepare_CommandQueue)
			Extrae_Probe_clEnqueueBarrier_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueBarrier != NULL)
	{
		r = real_clEnqueueBarrier (q);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueBarrier was not hooked!\n");
		exit (-1);
	}

	return r;
}

void *clEnqueueMapBuffer (cl_command_queue q, cl_mem m, cl_bool b,
	cl_map_flags mf, size_t o, size_t s, cl_uint n, const cl_event *ewl,
	cl_event *e, cl_int *err)
{
	void *r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueMapBuffer (real at %p)\n", real_clEnqueueMapBuffer);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueMapBuffer != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueMapBuffer_Enter ();
		r = real_clEnqueueMapBuffer (q, m, b, mf, o, s, n, ewl, &evt, err);
		Extrae_OpenCL_addEventToQueue (q, evt, OPENCL_CLENQUEUEMAPBUFFER_ACC_EV);
		if (e != NULL)
			*e = evt;
		if (b && !Extrae_OpenCL_Queue_OoO (q))
			Extrae_OpenCL_clQueueFlush (q, FALSE);
		Extrae_Probe_clEnqueueMapBuffer_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueMapBuffer != NULL)
	{
		r = real_clEnqueueMapBuffer (q, m, b, mf, o, s, n, ewl, e, err);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueMapBuffer was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clEnqueueUnmapMemObject (cl_command_queue q, cl_mem m, void *p,
	cl_uint n, const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueUnmapMemObject (real at %p)\n", real_clEnqueueUnmapMemObject);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueUnmapMemObject != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueUnmapMemObject_Enter ();
		r = real_clEnqueueUnmapMemObject (q, m, p, n, ewl, &evt);
		Extrae_OpenCL_addEventToQueue (q, evt, OPENCL_CLENQUEUEUNMAPMEMOBJECT_ACC_EV);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueUnmapMemObject_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueUnmapMemObject != NULL)
	{
		r = real_clEnqueueUnmapMemObject (q, m, p, n, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueUnmapMemObject was not hooked!\n");
		exit (-1);
	}

	return r;
}

#ifdef CL_VERSION_1_2
cl_int clEnqueueMigrateMemObjects (cl_command_queue q, cl_uint n, 
	const cl_mem *mo, cl_mem_migration_flags f, cl_uint ne,
	const cl_event *ewl, cl_event *e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clEnqueueUnmapMemObject (real at %p)\n", real_clEnqueueUnmapMemObject);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clEnqueueMigrateMemObjects != NULL)
	{
		cl_event evt;

		Extrae_Probe_clEnqueueMigrateMemObjects_Enter ();
		r = real_clEnqueueMigrateMemObjects (q, n, mo, f, ne, ewl, &evt);
		Extrae_OpenCL_addEventToQueue (q, evt, OPENCL_CLENQUEUEMIGRATEMEMOBJECTS_ACC_EV);
		if (e != NULL)
			*e = evt;
		Extrae_Probe_clEnqueueMigrateMemObjects_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clEnqueueMigrateMemObjects != NULL)
	{
		r = real_clEnqueueMigrateMemObjects (q, n, mo, f, ne, ewl, e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clEnqueueMigrateMemObjects was not hooked!\n");
		exit (-1);
	}

	return r;
}
#endif

cl_int clRetainCommandQueue (cl_command_queue cq)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clRetainCommandQueue (real at %p)\n", real_clRetainCommandQueue);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clRetainCommandQueue != NULL)
	{
		Extrae_Probe_clRetainCommandQueue_Enter ();
		r = real_clRetainCommandQueue (cq);
		Extrae_Probe_clRetainCommandQueue_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clRetainCommandQueue != NULL)
	{
		r = real_clRetainCommandQueue (cq);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainCommandQueue was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clReleaseCommandQueue (cl_command_queue cq)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clReleaseCommandQueue (real at %p)\n", real_clReleaseCommandQueue);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clReleaseCommandQueue != NULL)
	{
		Extrae_Probe_clReleaseCommandQueue_Enter ();
		r = real_clReleaseCommandQueue (cq);
		Extrae_Probe_clReleaseCommandQueue_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clReleaseCommandQueue != NULL)
	{
		r = real_clReleaseCommandQueue (cq);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clReleaseCommandQueue was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clRetainContext (cl_context c)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clRetainContext (real at %p)\n", real_clRetainContext);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clRetainContext != NULL)
	{
		Extrae_Probe_clRetainContext_Enter ();
		r = real_clRetainContext (c);
		Extrae_Probe_clRetainContext_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clRetainContext != NULL)
	{
		r = real_clRetainContext (c);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainContext was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clReleaseContext (cl_context c)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clReleaseContext (real at %p)\n", real_clRetainContext);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clReleaseContext != NULL)
	{
		Extrae_Probe_clReleaseContext_Enter ();
		r = real_clReleaseContext (c);
		Extrae_Probe_clReleaseContext_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clReleaseContext != NULL)
	{
		r = real_clReleaseContext (c);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clReleaseContext was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clRetainDevice (cl_device_id d)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clRetainDevice (real at %p)\n", real_clRetainDevice);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clRetainDevice != NULL)
	{
		Extrae_Probe_clRetainDevice_Enter ();
		r = real_clRetainDevice (d);
		Extrae_Probe_clRetainDevice_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clRetainDevice != NULL)
	{
		r = real_clRetainDevice (d);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainDevice was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clReleaseDevice (cl_device_id d)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clReleaseDevice (real at %p)\n", real_clReleaseDevice);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clReleaseDevice != NULL)
	{
		Extrae_Probe_clReleaseDevice_Enter ();
		r = real_clReleaseDevice (d);
		Extrae_Probe_clReleaseDevice_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clReleaseDevice != NULL)
	{
		r = real_clReleaseDevice (d);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clReleaseDevice was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int Extrae_clRetainEvent_real (cl_event e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : Extrae_clRetainEvent_real (real at %p)\n", real_clRetainEvent);
#endif

	if (real_clRetainEvent != NULL)
		r = real_clRetainEvent (e);
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainEvent was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int Extrae_clReleaseEvent_real (cl_event e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : Extrae_clReleaseEvent_real (real at %p)\n", real_clReleaseEvent);
#endif

	if (real_clReleaseEvent != NULL)
		r = real_clReleaseEvent (e);
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clReleaseEvent was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clRetainEvent (cl_event e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clRetainEvent (real at %p)\n", real_clRetainEvent);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clRetainEvent != NULL)
	{
		Extrae_Probe_clRetainEvent_Enter ();
		r = real_clRetainEvent (e);
		Extrae_Probe_clRetainEvent_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clRetainEvent != NULL)
	{
		r = real_clRetainEvent (e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainEvent was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clReleaseEvent (cl_event e)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clReleaseEvent (real at %p)\n", real_clReleaseEvent);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clReleaseEvent != NULL)
	{
		Extrae_Probe_clReleaseEvent_Enter ();
		r = real_clReleaseEvent (e);
		Extrae_Probe_clReleaseEvent_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clReleaseEvent != NULL)
	{
		r = real_clReleaseEvent (e);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clReleaseEvent was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clRetainKernel (cl_kernel k)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clRetainKernel (real at %p)\n", real_clRetainKernel);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clRetainKernel != NULL)
	{
		Extrae_Probe_clRetainKernel_Enter ();
		r = real_clRetainKernel (k);
		Extrae_Probe_clRetainKernel_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clRetainKernel != NULL)
	{
		r = real_clRetainKernel (k);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainKernel was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clReleaseKernel (cl_kernel k)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clReleaseKernel (real at %p)\n", real_clReleaseKernel);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clReleaseKernel != NULL)
	{
		Extrae_Probe_clReleaseKernel_Enter ();
		r = real_clReleaseKernel (k);
		Extrae_Probe_clReleaseKernel_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clReleaseKernel != NULL)
	{
		r = real_clReleaseKernel (k);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRelaseKernel was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clRetainMemObject (cl_mem m)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clRetainMemObject (real at %p)\n", real_clRetainMemObject);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clRetainMemObject != NULL)
	{
		Extrae_Probe_clRetainMemObject_Enter ();
		r = real_clRetainMemObject (m);
		Extrae_Probe_clRetainMemObject_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clRetainMemObject != NULL)
	{
		r = real_clRetainMemObject (m);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainMemObject was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clReleaseMemObject (cl_mem m)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clReleaseMemObject (real at %p)\n", real_clReleaseMemObject);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clReleaseMemObject != NULL)
	{
		Extrae_Probe_clReleaseMemObject_Enter ();
		r = real_clReleaseMemObject (m);
		Extrae_Probe_clReleaseMemObject_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clReleaseMemObject != NULL)
	{
		r = real_clReleaseMemObject (m);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clReleaseMemObject was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clRetainProgram (cl_program p)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clRetainProgram (real at %p)\n", real_clRetainProgram);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clRetainProgram != NULL)
	{
		Extrae_Probe_clRetainProgram_Enter ();
		r = real_clRetainProgram (p);
		Extrae_Probe_clRetainProgram_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clRetainProgram != NULL)
	{
		r = real_clRetainProgram (p);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clRetainProgram was not hooked!\n");
		exit (-1);
	}

	return r;
}

cl_int clReleaseProgram (cl_program p)
{
	cl_int r;

#ifdef DEBUG
	fprintf (stderr, PACKAGE_NAME": Debug : clReleaseProgram (real at %p)\n", real_clReleaseProgram);
#endif

	if (EXTRAE_ON() && Extrae_get_trace_OpenCL() && real_clReleaseProgram != NULL)
	{
		Extrae_Probe_clReleaseProgram_Enter ();
		r = real_clReleaseProgram (p);
		Extrae_Probe_clReleaseProgram_Exit ();
	}
	else if (!(EXTRAE_ON() && Extrae_get_trace_OpenCL()) && real_clReleaseProgram != NULL)
	{
		r = real_clReleaseProgram (p);
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": Fatal Error! clReleaseProgram was not hooked!\n");
		exit (-1);
	}

	return r;
}

#endif /* PIC */
