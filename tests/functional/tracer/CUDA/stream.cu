#include <cuda.h>
#include <stdio.h>

__global__ void helloStream(char *);

void do_work (void)
{
	int i;

	// desired output
	char str[] = "Hello World!";

	for(i = 0; i < 12; i++)
		str[i] -= i;

	cudaStream_t s1;
	cudaStreamCreateWithFlags(&s1, cudaStreamDefault);

	// allocate memory on the device
	char *d_str;
	size_t size = sizeof(str);
	cudaMalloc((void**)&d_str, size);

	// copy the string to the device using the stream
	cudaMemcpyAsync(d_str, str, size, cudaMemcpyHostToDevice, s1);

	// set the grid and block sizes
	dim3 dimGrid(2);   // one block per word
	dim3 dimBlock(6); // one thread per character

	// invoke the kernel on the stream
	helloStream<<< dimGrid, dimBlock, 0, s1 >>>(d_str);

	// retrieve the results from the device
	cudaMemcpyAsync(str, d_str, size, cudaMemcpyDeviceToHost, s1);

	// Sync all threads
	cudaThreadSynchronize();

	// free up the allocated memory on the device
	cudaFree(d_str);

	printf("%s\n", str);

	cudaStreamDestroy(s1);
}

int main (int argc, char *argv[])
{
	do_work();

	return 0;
}

// Device kernel
__global__ void helloStream(char* str)
{
	// determine where in the thread grid we are
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// unmangle output
	str[idx] += idx;
}
