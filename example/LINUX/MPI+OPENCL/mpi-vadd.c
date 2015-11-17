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

//------------------------------------------------------------------------------
//
// Name:       vadd.c
// 
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Harald Servat, June 2013
//             
//------------------------------------------------------------------------------

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef APPLE
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include "CL/cl.h"
#endif
#include <mpi.h>

//pick up device type from compiler command line or from 
//the default type
#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_GPU
#endif

static int output_device_info (int mpirank, cl_device_id device_id)
{
	int err;                            // error code returned from OpenCL calls
	cl_device_type device_type;         // Parameter defining the type of the compute device
	cl_uint comp_units;                 // the max number of compute units on a device
	cl_char vendor_name[1024] = {0};    // string to hold vendor name for compute device
	cl_char device_name[1024] = {0};    // string to hold name of compute device


	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to access device name!\n");
		return EXIT_FAILURE;
	}
	printf("MPI rank %d : Device is  %s ", mpirank, device_name);

	err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to access device type information!\n");
		return EXIT_FAILURE;
	}
	if(device_type  == CL_DEVICE_TYPE_GPU)
		printf(" GPU from ");
	else if (device_type == CL_DEVICE_TYPE_CPU)
		printf("\n CPU from ");
	else 
		printf("\n non  CPU or GPU processor from ");

	err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to access device vendor name!\n");
		return EXIT_FAILURE;
	}
	printf(" %s ",vendor_name);

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to access device number of compute units !\n");
		return EXIT_FAILURE;
	}
	printf(" with a max of %d compute units\n",comp_units);

	return CL_SUCCESS;
}

//------------------------------------------------------------------------------

#define TOL    (0.001)   // tolerance used in floating point comparisons
#define LENGTH (1024)    // length of vectors a, b, and c

//------------------------------------------------------------------------------
//
// kernel:  vadd  
//
// Purpose: Compute the elementwise sum c = a+b
// 
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//
 
const char *KernelSource = "\n" \
"__kernel void vadd(                                                 \n" \
"   __global float* a,                                                  \n" \
"   __global float* b,                                                  \n" \
"   __global float* c,                                                  \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       c[i] = a[i] + b[i];                                             \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
	int          rank, size;         // MPI rank & size
	int          err;                // error code returned from OpenCL calls
	float        h_a[LENGTH];        // a vector 
	float        h_b[LENGTH];        // b vector 
	float        h_c[LENGTH];        // c vector (a+b) returned from the compute device (local per task)
	float        _h_c[LENGTH];       // c vector (a+b) returned from the compute device (global for master)
	unsigned int correct;            // number of correct results  

	size_t global;                   // global domain size  
	size_t local;                    // local  domain size  

	cl_device_id     device_id;      // compute device id 
	cl_context       context;        // compute context
	cl_command_queue commands;       // compute command queue
	cl_program       program;        // compute program
	cl_kernel        ko_vadd;        // compute kernel
    
	cl_mem d_a;                      // device memory used for the input  a vector
	cl_mem d_b;                      // device memory used for the input  b vector
	cl_mem d_c;                      // device memory used for the output c vector

	int mycount, i;

	err = MPI_Init (&argc, &argv);

	if (err != MPI_SUCCESS)
	{
		printf ("MPI_Init failed!\n");
		exit (-1);
	}

	err = MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	if (err != MPI_SUCCESS)
	{
		printf ("MPI_Comm_rank failed!\n");
		exit (-1);
	}

	err = MPI_Comm_size (MPI_COMM_WORLD, &size);
	if (err != MPI_SUCCESS)
	{
		printf ("MPI_Comm_size failed\n");
		exit (-1);
	}

	if (LENGTH % size != 0)
	{
		printf ("Number of MPI processes must divide LENGTH (%d)\n", LENGTH);
		exit (-1);
	}

	mycount = LENGTH / size;
    
	if (rank == 0)
	{
		for (i = 0; i < LENGTH; i++)
		{
			h_a[i] = rand() / (float)RAND_MAX;
			h_b[i] = rand() / (float)RAND_MAX;
			h_a[i] = i;
			h_b[i] = i*2;
		}
		err = MPI_Bcast (h_a, LENGTH, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS)
		{
			printf ("MPI_Bcast failed transferring h_a\n");
			exit (-1);
		}
		err = MPI_Bcast (h_b, LENGTH, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS)
		{
			printf ("MPI_Bcast failed transferring h_b\n");
			exit (-1);
		}
	}
	else
	{
		err = MPI_Bcast (h_a, LENGTH, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS)
		{
			printf ("MPI_Bcast failed receiving h_a\n");
			exit (-1);
		}
		err = MPI_Bcast (h_b, LENGTH, MPI_FLOAT, 0, MPI_COMM_WORLD);
		if (err != MPI_SUCCESS)
		{
			printf ("MPI_Bcast failed receiving h_b\n");
			exit (-1);
		}
	}
    
	// Set up platform 
	cl_uint numPlatforms;

	// Find number of platforms
	err = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to find a platform!\n");
		return EXIT_FAILURE;
	}

	// Get all platforms
	cl_platform_id Platform[numPlatforms];
	err = clGetPlatformIDs(numPlatforms, Platform, NULL);
	if (err != CL_SUCCESS || numPlatforms <= 0)
	{
		printf("Error: Failed to get the platform!\n");
		return EXIT_FAILURE;
	}

	// Secure a GPU
	for (i = 0; i < numPlatforms; i++)
	{
		err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
		if (err == CL_SUCCESS)
			break;
	}

	if (device_id == NULL)
	{
		printf("Error: Failed to create a device group!\n");
		return EXIT_FAILURE;
	}
	else
	{
		if (output_device_info (rank, device_id) != CL_SUCCESS)
			return EXIT_FAILURE;
	}

	// Create a compute context 
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (!context)
	{
		printf("Error: Failed to create a compute context!\n");
		return EXIT_FAILURE;
	}

	// Create a command queue
	commands = clCreateCommandQueue(context, device_id, 0, &err);
	if (!commands)
	{
		printf("Error: Failed to create a command commands!\n");
		return EXIT_FAILURE;
	}

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
	if (!program)
	{
		printf("Error: Failed to create compute program!\n");
		return EXIT_FAILURE;
	}

	// Build the program  
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t len;
		char buffer[2048];

		printf("Error: Failed to build program executable!\n");
		clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	// Create the compute kernel from the program 
	ko_vadd = clCreateKernel(program, "vadd", &err);
	if (!ko_vadd || err != CL_SUCCESS)
	{
		printf("Error: Failed to create compute kernel!\n");
		exit(1);
	}

	// Create the input (a, b) and output (c) arrays in device memory  
	d_a = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * mycount, NULL, NULL);
	d_b = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * mycount, NULL, NULL);
	d_c = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * mycount, NULL, NULL);
	if (!d_a || !d_b || !d_c)
	{
		printf("Error: Failed to allocate device memory!\n");
		exit(1);
	}    
    
	// Write a and b vectors into compute device memory 
	err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * mycount, &h_a[rank*mycount], 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write h_a to source array!\n");
		exit(1);
	}

	err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * mycount, &h_b[rank*mycount], 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to write h_b to source array!\n");
		exit(1);
	}
	
	// Set the arguments to our compute kernel
	err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
	err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
	err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
	err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &mycount);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to set kernel arguments! %d\n", err);
		exit(1);
	}

	// Get the maximum work group size for executing the kernel on the device
	err = clGetKernelWorkGroupInfo(ko_vadd, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to retrieve kernel work group info! %d\n", err);
		exit(1);
	}

	// Execute the kernel over the entire range of our 1d input data set
	// using the maximum number of work group items for this device
	global = LENGTH;
	err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, &local, 0, NULL, NULL);
	if (err)
	{
		printf("Error: Failed to execute kernel!\n");
		return EXIT_FAILURE;
	}

	// Wait for the commands to complete before reading back results
	clFinish(commands);

	// Read back the results from the compute device
	err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * mycount, &h_c, 0, NULL, NULL );  
	if (err != CL_SUCCESS)
	{
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	err = MPI_Gather (h_c, mycount, MPI_FLOAT, _h_c, mycount, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if (err != MPI_SUCCESS)
	{
		printf ("MPI_Gather failed receiving h_c\n");
		exit (-1);
	}

	if (rank == 0)
	{
		// Test the results
		correct = 0;
		float tmp;
    
		for(i = 0; i < LENGTH; i++)
		{
			tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
			tmp -= _h_c[i];             // compute deviation of expected and output result
			if(tmp*tmp < TOL*TOL)      // correct if square deviation is less than tolerance squared
				correct++;
			else 
				printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], _h_c[i]);
		}

		// summarize results
		printf("C = A+B:  %d out of %d results were correct.\n", correct, LENGTH);
	}
    
	// cleanup then shutdown
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_c);
	clReleaseProgram(program);
	clReleaseKernel(ko_vadd);
	clReleaseCommandQueue(commands);
	clReleaseContext(context);

	err = MPI_Finalize ();
	if (err != MPI_SUCCESS)
	{
		printf ("MPI_Finalize failed!\n");
		exit (-1);
	}

	return 0;
}

