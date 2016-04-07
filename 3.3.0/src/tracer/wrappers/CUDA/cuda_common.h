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

#include <config.h>

#if CUDA_WITHOUT_CUPTI_INSTRUMENTATION

/**
 ** The following lines are convenient hacks to avoid including cupti.h
 **/

typedef int cudaError_t;
enum cudaMemcpyKind
{
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};
typedef struct dim3_st { unsigned int x, y, z; } dim3;
typedef void * cudaEvent_t;
typedef void * cudaStream_t;
typedef enum CUevent_flags_enum { CU_EVENT_DEFAULT, CU_EVENT_BLOCKING_SYNC, CU_EVENT_DISABLE_TIMING } CUevent_flags;

/* structures defined within generated_cuda_runtime_api_meta.h */

typedef struct cudaLaunch_v3020_params_st {
	const char *entry;
} cudaLaunch_v3020_params;

typedef struct cudaConfigureCall_v3020_params_st {
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
} cudaConfigureCall_v3020_params;

typedef struct cudaStreamCreate_v3020_params_st {
	cudaStream_t *pStream;
} cudaStreamCreate_v3020_params;

typedef struct cudaStreamSynchronize_v3020_params_st {
	cudaStream_t stream;
} cudaStreamSynchronize_v3020_params;

typedef struct cudaMemcpy_v3020_params_st {
	void *dst;
	const void *src;
	size_t count;
	enum cudaMemcpyKind kind;
} cudaMemcpy_v3020_params;

typedef struct cudaMemcpyAsync_v3020_params_st {
	void *dst;
	const void *src;
	size_t count;
	enum cudaMemcpyKind kind;
	cudaStream_t stream;
} cudaMemcpyAsync_v3020_params;


/* From cuda_runtime_api.h */

cudaError_t cudaGetDevice (int *);
cudaError_t cudaEventCreateWithFlags (cudaEvent_t *, unsigned);
cudaError_t cudaEventRecord(cudaEvent_t , cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaGetDeviceCount(int *);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);

#else /* CUDA_WITHOUT_CUPTI_INSTRUMENTATION */

# include <cuda_runtime_api.h>
# include <cupti.h>
# include <cupti_events.h>

#endif /* CUDA_WITHOUT_CUPTI_INSTRUMENTATION */

#define MAX_CUDA_EVENTS 32768

#define CUDA_SUCCESS 0

#define CHECK_CU_ERROR(err, cufunc)                             \
  if (err != CUDA_SUCCESS)                                      \
    {                                                           \
      printf ("Error %d for CUDA Driver API function '%s'.\n",  \
              err,  #cufunc);                                   \
      exit(-1);                                                 \
    }

typedef enum {
	EXTRAE_CUDA_NEW_TIME,
	EXTRAE_CUDA_PREVIOUS_TIME
} Extrae_CUDA_Time_Type;

/* Information per stream required during tracing */
struct RegisteredStreams_t
{
	UINT64 host_reference_time;
	cudaEvent_t device_reference_time; /* accessed through cudaEvent_t */
	unsigned threadid; /* In Paraver sense */
	cudaStream_t stream;

	unsigned nevents;
	cudaEvent_t ts_events[MAX_CUDA_EVENTS];
	unsigned events[MAX_CUDA_EVENTS];
	unsigned long long values[MAX_CUDA_EVENTS];
	unsigned tag[MAX_CUDA_EVENTS];
	unsigned size[MAX_CUDA_EVENTS];
	Extrae_CUDA_Time_Type timetype[MAX_CUDA_EVENTS];
};

struct CUDAdevices_t
{
	struct RegisteredStreams_t *Stream;
	int nstreams;
	int initialized;
#if 0
	/* To perform sampling, CUPTI */
	CUcontext context;
	CUdevice device;
#endif
};

void Extrae_CUDA_flush_all_streams(int synchronize);
void Extrae_cudaLaunch_Enter (const char*);
void Extrae_cudaLaunch_Exit (void);
void Extrae_cudaConfigureCall_Enter (dim3, dim3, size_t, cudaStream_t);
void Extrae_cudaConfigureCall_Exit (void);
void Extrae_cudaThreadSynchronize_Enter (void);
void Extrae_cudaThreadSynchronize_Exit (void);
void Extrae_cudaDeviceSynchronize_Enter (void);
void Extrae_cudaDeviceSynchronize_Exit (void);
void Extrae_cudaStreamCreate_Enter (cudaStream_t*);
void Extrae_cudaStreamCreate_Exit (void);
void Extrae_cudaStreamSynchronize_Enter (cudaStream_t);
void Extrae_cudaStreamSynchronize_Exit (void);
void Extrae_cudaMemcpy_Enter (void*, const void*, size_t, enum cudaMemcpyKind);
void Extrae_cudaMemcpy_Exit (void);
void Extrae_cudaMemcpyAsync_Enter (void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);
void Extrae_cudaMemcpyAsync_Exit (void);
void Extrae_cudaDeviceReset_Enter (void);
void Extrae_cudaDeviceReset_Exit (void);
void Extrae_cudaThreadExit_Enter (void);
void Extrae_cudaThreadExit_Exit (void);

void Extrae_reallocate_CUDA_info (unsigned nthreads);

void Extrae_CUDA_fini (void);
