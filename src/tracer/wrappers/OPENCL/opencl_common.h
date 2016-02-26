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

#ifndef OPENCL_COMMON_H_INCLUDED
#define OPENCL_COMMON_H_INCLUDED

#include <CL/cl.h>

#include <opencl_wrapper.h>

void Extrae_OpenCL_clCreateCommandQueue (cl_command_queue queue,
	cl_device_id device, cl_command_queue_properties properties);

int Extrae_OpenCL_Queue_OoO (cl_command_queue q);

void Extrae_OpenCL_clQueueFlush (cl_command_queue queue, int addFinish);
void Extrae_OpenCL_clQueueFlush_All (void);

void Extrae_OpenCL_addEventToQueue (cl_command_queue queue, cl_event ocl_evt, 
	unsigned prv_evt);
void Extrae_OpenCL_addEventToQueueWithKernel (cl_command_queue queue,
	cl_event ocl_evt, unsigned prv_evt, cl_kernel k);
void Extrae_OpenCL_addEventToQueueWithSize (cl_command_queue queue, cl_event ocl_evt, 
	unsigned prv_evt, size_t size);

void Extrae_OpenCL_annotateKernelName (cl_kernel k, unsigned *pos);

unsigned Extrae_OpenCL_tag_generator (void);

unsigned Extrae_OpenCL_lookForOpenCLQueueToThreadID (cl_command_queue q);


#endif /* OPENCL_WRAPPER_H_INCLUDED */
