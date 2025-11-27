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

#include "hip_common.h"

#include "taskid.h"
#include "threadinfo.h"
#include "wrapper.h"
#include "trace_macros.h"
#include "trace_macros_gpu.h"
#include "hip_probe.h"
#include "xalloc.h"
#include "extrae_user_events.h"
#include "gpu_event_info.h"
#include "trace_mode.h"

#if defined(__APPLE__)
# define HOST_NAME_MAX 512
#endif

#ifndef HI_STREAM_LEGACY
#define HI_STREAM_LEGACY ((hipStream_t)(uintptr_t)-1)
#endif

#ifndef HI_STREAM_PER_THREAD
#define HI_STREAM_PER_THREAD ((hipStream_t)(uintptr_t)-2)
#endif

#ifndef HIP_MAP_NUM_BUCKETS
#define HIP_MAP_NUM_BUCKETS 4096u
#endif

/* Structures that will hold the parameters needed for the exit parts of the
 * instrumentation code. This way we can support dyninst/ld-preload/cupti
 * instrumentation with a single file
 */

typedef struct 
{
	hipStream_t *stream;
} hipStreamCreate_saved_params_t;

typedef struct
{
	hipStream_t stream;
} hipStreamSynchronize_saved_params_t;

typedef struct
{
	size_t size;
	enum hipMemcpyKind kind;
} hipMemcpy_saved_params_t;

typedef struct
{
	size_t size;
	enum hipMemcpyKind kind;
	hipStream_t stream;
} hipMemcpyAsync_saved_params_t;

typedef union
{
	hipStreamCreate_saved_params_t csc;
	hipStreamSynchronize_saved_params_t css;
	hipMemcpy_saved_params_t cm;
	hipMemcpyAsync_saved_params_t cma;
} Extrae_hip_saved_params_union;

typedef struct
{
	int instrumentationDepth;
	Extrae_hip_saved_params_union punion;
} Extrae_hip_saved_params_t;

unsigned hip_events_block_size = DEFAULT_HIP_EVENTS_BLOCK_SIZE;

static Extrae_hip_saved_params_t *Extrae_HIP_saved_params = NULL;

static unsigned __last_tag = 0xC0DA; /* Fixed tag */
static unsigned Extrae_HIP_tag_generator(void)
{
	__last_tag++;
	return __last_tag;
}

static unsigned Extrae_HIP_tag_get()
{
	return __last_tag;
}

static struct HIPdevices_t *devices = NULL;
static int HIPdevices = 0;

/**
 * Variable to control the initialization and allow the flushing of
 * events at the exit point.
 */
int hipInitialized = 0;

gpu_event_list_t availableEvents;

// Array of hash buckets for the kernel map.
// Each bucket is the head of a linked list of KernelMapEntry nodes.
KernelMapEntry* kernel_map_buckets[HIP_MAP_NUM_BUCKETS] = {0};

// Read–write lock protecting the entire kernel map.
// Readers can access concurrently, writers get exclusive access.
static pthread_rwlock_t kernel_map_rwlock = PTHREAD_RWLOCK_INITIALIZER;

void Extrae_HIP_updateDepth_(int step)
{
	if (Extrae_HIP_saved_params == NULL) Extrae_reallocate_HIP_info(0, Backend_getMaximumOfThreads());
	Extrae_HIP_saved_params[THREADID].instrumentationDepth += step;
}

int Extrae_HIP_getDepth()
{
	if (Extrae_HIP_saved_params == NULL) return 0;
	return Extrae_HIP_saved_params[THREADID].instrumentationDepth;
}

const char* NormalizeKernelName(const char* name)
{
    static char buffer[512];
    const char* p = strchr(name, '$');
    if (p) {
        size_t len = p - name;
        if (len >= sizeof(buffer)) len = sizeof(buffer) - 1;
        strncpy(buffer, name, len);
        buffer[len] = '\0';
        return buffer;
    }
    return name;
}

static void Extrae_HIP_SynchronizeStream(int devid, int streamid)
{
	int err;

	if (devid >= HIPdevices)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Invalid HIP device id in HIPSynchronizeStream\n");
		exit (-1);
	}

#ifdef DEBUG
	fprintf (stderr, "Extrae_HIP_SynchronizeStream (devid=%d, streamid=%d, stream=%p)\n", devid, streamid, devices[devid].Stream[streamid].stream);
#endif

	if(devices[devid].Stream[streamid].device_reference_event == NULL)
	{
		devices[devid].Stream[streamid].device_reference_event = gpuEventList_pop(&devices[devid].availableEvents);
	}

	err = hipEventRecord (devices[devid].Stream[streamid].device_reference_event->ts_event, devices[devid].Stream[streamid].stream);
	CHECK_HI_ERROR(err, hipEventRecord);

	err = hipEventSynchronize (devices[devid].Stream[streamid].device_reference_event->ts_event);
	CHECK_HI_ERROR(err, hipEventSynchronize);

	devices[devid].Stream[streamid].host_reference_time = TIME;
}

void Extrae_HIP_deInitialize(int devid)
{
	if (devices != NULL)
	{
		if (devices[devid].initialized)
		{
			xfree (devices[devid].Stream);
			devices[devid].Stream = NULL;
			devices[devid].initialized = FALSE;
		}
	}
}

void Extrae_HIP_Initialize(int devid)
{
	hipError_t err;
	int i;

	/* If devices table is not initialized, create it first */
	if (devices == NULL)
	{
		err = hipGetDeviceCount (&HIPdevices);
		CHECK_HI_ERROR(err, hipGetDeviceCount);

		devices = (struct HIPdevices_t*) xmalloc (sizeof(struct HIPdevices_t)*HIPdevices);

		for (i = 0; i < HIPdevices; i++)
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

		devices[devid].Stream = (struct RegisteredStreams_t*) xmalloc_and_zero (devices[devid].nstreams*sizeof(struct RegisteredStreams_t));

		/* Was the thread created before (i.e. did we executed a hipdevicereset?) */
		if (gethostname(_hostname, HOST_NAME_MAX) == 0)
			sprintf (_threadname, "HIP-D%d.S%d-%s", devid+1, 1, _hostname);
		else
			sprintf (_threadname, "HIP-D%d.S%d-%s", devid+1, 1, "unknown-host");
		prev_threadid = Extrae_search_thread_name (_threadname, &found);

		if (found)
		{
			/* If thread name existed, reuse its thread id */
			devices[devid].Stream[0].threadid = prev_threadid;
		}
		else
		{
			/* For timing purposes we change num of threads here instead of doing Backend_getNumberOfThreads() + HIPdevices*/
			/*
			 * XXX Should this be Backend_getMaximumOfThreads()? If we
			 * previously increased the number of threads in another runtime,
			 * and then decreased them, we will end up with a line with mixed
			 * semantics (thread&stream).
			 */
			Backend_ChangeNumberOfThreads(Backend_getNumberOfThreads() + 1);
			devices[devid].Stream[0].threadid = Backend_getNumberOfThreads()-1;

			/* Set thread name */
			Extrae_set_thread_name(devices[devid].Stream[0].threadid, _threadname);
		}

		/* default device stream */
		devices[devid].Stream[0].stream = (hipStream_t) 0;
		gpuEventList_init(&devices[devid].Stream[0].gpu_event_list, FALSE, XTR_HIP_EVENTS_BLOCK_SIZE);
		gpuEventList_init(&devices[devid].availableEvents, TRUE, XTR_HIP_EVENTS_BLOCK_SIZE);

		/* Create an event record and process it through the stream! */
		/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */

		devices[devid].Stream[0].device_reference_event = NULL;
		Extrae_HIP_SynchronizeStream (devid, 0);

		/*
		 * Necessary to change the base state of HIP streams from NOT_TRACING
		 * to IDLE. We manually emit a TRACING_MODE_DETAIL event because HIP
		 * doesn't have burst mode yet. Will need a revision when supported.
		 */
		THREAD_TRACE_MISCEVENT(devices[devid].Stream[0].threadid, devices[devid].Stream[0].host_reference_time, TRACING_MODE_EV, TRACE_MODE_DETAIL, 0);

		devices[devid].initialized = TRUE;

		/**
		* Last flush of stream events.
		* This initialization occurs in the callback of a HIP call from the app,
		* therefore this atexit corresponds to the exit point of the main from the app traced.
		* Moving this to CUPTI initialization is not equivalent since that initialization
		* is called from the Extrae constructor, and therefore that atexit corresponds to our library exit point.
		*/
		atexit(Extrae_HIP_finalize);
		hipInitialized = 1;
	}
}

static int Extrae_HIP_SearchStream(int devid, hipStream_t stream)
{
	int i;

	/* Starting from HIP 8, CU_STREAM_LEGACY is a new stream handle that uses 
	   an implicit stream with legacy synchronization behavior, just as the 
	   behaviour of stream 0 (default).

		 HI_STREAM_PER_THREAD is assigned to the tid 0. This is allow apps that used it
		 to generate a trace but this could lead to overlapped kernel events if 
		 several threads of the same parallel use HI_STREAM_PER_THREAD concurrently.
	 */
	if (stream == HI_STREAM_LEGACY || stream == HI_STREAM_PER_THREAD) return 0;

	for (i = 0; i < devices[devid].nstreams; i++)
		if (devices[devid].Stream[i].stream == stream)
			return i;

	return -1;
}

static void Extrae_HIP_unRegisterStream(int devid, hipStream_t stream)
{
	int stid = Extrae_HIP_SearchStream (devid, stream);

#ifdef DEBUG
	fprintf (stderr, "Extrae_HIP_unRegisterStream (devid=%d, stream=%p unassigned from streamid => %d/%d\n", devid, stream, stid, devices[devid].nstreams);
#endif

	Extrae_HIP_flush_streams(devid, stid);

	int nstreams = devices[devid].nstreams - 1;

	struct RegisteredStreams_t *rs_tmp = (struct RegisteredStreams_t*) xmalloc (nstreams*sizeof(struct RegisteredStreams_t));

	memmove (rs_tmp, devices[devid].Stream, stid * sizeof(struct RegisteredStreams_t));
	memmove (rs_tmp+stid, devices[devid].Stream + stid + 1, (devices[devid].nstreams - stid - 1)*sizeof(struct RegisteredStreams_t));

	devices[devid].nstreams = nstreams;

	xfree (devices[devid].Stream);
	devices[devid].Stream = rs_tmp;
}

static void Extrae_HIP_RegisterStream(int devid, hipStream_t stream)
{
	int i = devices[devid].nstreams;

	devices[devid].Stream = (struct RegisteredStreams_t*) xrealloc (
	  devices[devid].Stream, (i+1)*sizeof(struct RegisteredStreams_t));

	devices[devid].nstreams++;

	/*
	 * XXX Should this be Backend_getMaximumOfThreads()? If we
	 * previously increased the number of threads in another runtime,
	 * and then decreased them, we will end up with a line with mixed
	 * semantics (thread&stream).
	 */
	Backend_ChangeNumberOfThreads(Backend_getNumberOfThreads()+1);

	devices[devid].Stream[i].host_reference_time = 0;
	devices[devid].Stream[i].device_reference_event = NULL;
	devices[devid].Stream[i].threadid = Backend_getNumberOfThreads()-1;
	devices[devid].Stream[i].stream = stream;
	gpuEventList_init(&devices[devid].Stream[i].gpu_event_list, FALSE, XTR_HIP_EVENTS_BLOCK_SIZE);

#ifdef DEBUG
	fprintf(stderr, "Extrae_HIP_RegisterStream (devid=%d, stream=%p assigned to streamid => %d\n", devid, stream, i);
#endif

	/* Set thread name */
	{
		char _threadname[THREAD_INFO_NAME_LEN];
		char _hostname[HOST_NAME_MAX];

		if (gethostname(_hostname, HOST_NAME_MAX) == 0)
			sprintf(_threadname, "HIP-D%d.S%d-%s", devid+1, i+1, _hostname);
		else
			sprintf(_threadname, "HIP-D%d.S%d-%s", devid+1, i+1, "unknown-host");
		Extrae_set_thread_name(devices[devid].Stream[i].threadid, _threadname);
	}

	/* Create an event record and process it through the stream! */
	/* FIX CU_EVENT_BLOCKING_SYNC may be harmful!? */
	devices[devid].Stream[i].device_reference_event = NULL;
	Extrae_HIP_SynchronizeStream(devid, i);

	/*
	 * Necessary to change the base state of HIP streams from NOT_TRACING
	 * to IDLE. We manually emit a TRACING_MODE_DETAIL event because HIP
	 * doesn't have burst mode yet. Will need a revision when supported.
	 */
	THREAD_TRACE_MISCEVENT(devices[devid].Stream[i].threadid, devices[devid].Stream[i].host_reference_time, TRACING_MODE_EV, TRACE_MODE_DETAIL, 0);
}

static void Extrae_HIP_AddEventToStream(Extrae_HIP_Time_Type timetype, int devid, int streamid, unsigned event, unsigned long long value, unsigned tag, size_t size, unsigned blockspergrid, unsigned threadsperblock)
{
	int err;
	struct RegisteredStreams_t* registered_stream = &devices[devid].Stream[streamid];

	gpu_event_t* gpu_event = gpuEventList_pop(&devices[devid].availableEvents);

	gpu_event->timetype = timetype;
	gpu_event->event = event;
	gpu_event->value = value;
	gpu_event->tag = tag;
	gpu_event->memSize = size;
	gpu_event->blocksPerGrid = blockspergrid;
	gpu_event->threadsPerBlock = threadsperblock;

	err = hipEventRecord(gpu_event->ts_event, registered_stream->stream);
	CHECK_HI_ERROR(err, hipEventRecord);

	gpuEventList_add(&registered_stream->gpu_event_list, gpu_event);
}

/**
 * This routine emits GPU events into the tracing buffer and
 * inserts communication events to connect host-side 
 * HIP runtime calls with their corresponding GPU kernel and memcopies executions.
 *
 * @param threadid          buffer thread id
 * @param time              Timestamp of the event
 * @param event             kind of event (HIPKERNEL_GPU_VAL, HIPMEMCPY_GPU_VAL...)
 * @param value             Entry/exit (EVT_BEGIN/EVT_END)
 * @param tag               Communication tag
 * @param size              Size in bytes for memory operations
 * @param blockspergrid     Grid configuration (HIP kernels)
 * @param threadsperblock   Block configuration (HIP kernels)
 */
static void traceGPUEvents(int threadid, UINT64 time, unsigned event, unsigned long long value, unsigned tag, size_t size, unsigned blockspergrid, unsigned threadsperblock)
{
	TRACE_GPU_EVENT(threadid, time, HIPCALLGPU_EV, event, value, size);

	if (event == HIPKERNEL_GPU_VAL)
	{
		TRACE_GPU_KERNEL_EVENT(threadid, time, HIP_KERNEL_EXEC_EV, value, blockspergrid, threadsperblock);
		if(value != EVT_END) 
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, time,USER_RECV_EV,TASKID,0, tag, tag);
		}
	}
	else
	{
		if ((event == HIPMEMCPY_GPU_VAL || event == HIPMEMCPYASYNC_GPU_VAL) && (tag > 0))
		{
			THREAD_TRACE_USER_COMMUNICATION_EVENT(threadid, time, (value==EVT_END)?USER_RECV_EV:USER_SEND_EV, TASKID, size, tag, tag);
		}
	}
}

static void flushGPUEvents(int devid, int streamid)
{
	int threadid = devices[devid].Stream[streamid].threadid;
	int err;
	UINT64 utmp, last_time = 0;
	float ftmp;
	gpu_event_t *gpu_event;
	hipEvent_t *reference_event;
	UINT64 reference_time = 0;
	struct RegisteredStreams_t *registered_stream = &devices[devid].Stream[streamid];

	gpu_event_t *last_gpu_event = gpuEventList_peek_tail(&registered_stream->gpu_event_list);

	if(last_gpu_event != NULL)
	{
		err = hipEventSynchronize(last_gpu_event->ts_event);
		CHECK_HI_ERROR(err, hipEventSynchronize);
		reference_event = &registered_stream->device_reference_event->ts_event;
		reference_time = registered_stream->host_reference_time;

		/* Translate time from GPU to CPU using .device_reference_time and .host_reference_time
				from the RegisteredStreams_t structure */

		gpu_event = gpuEventList_pop(&registered_stream->gpu_event_list);
		while(gpu_event != NULL)
		{
			if (gpu_event->timetype == EXTRAE_HIP_NEW_TIME)
			{
				err = hipEventElapsedTime (&ftmp, *reference_event, gpu_event->ts_event);
				CHECK_HI_ERROR(err, hipEventElapsedTime);
				ftmp *= 1000000;
				/* Time correction between CPU & GPU */
				utmp = reference_time + (UINT64) (ftmp);

				reference_event = &gpu_event->ts_event;
				reference_time = utmp;
			}
			else
			{
				utmp = last_time;
			}

			traceGPUEvents(threadid, utmp, gpu_event->event, gpu_event->value, gpu_event->tag, gpu_event->memSize, 0, 0);

			last_time = utmp;

			if(gpuEventList_isempty(&registered_stream->gpu_event_list))
			{
				gpuEventList_add(&devices[devid].availableEvents, registered_stream->device_reference_event);
				registered_stream->device_reference_event = gpu_event;
				registered_stream->host_reference_time = last_time;
				gpu_event = NULL;
			}
			else
			{
				gpuEventList_add(&devices[devid].availableEvents, gpu_event);
				gpu_event = gpuEventList_pop(&registered_stream->gpu_event_list);
			}
		}
	}
}

/** 
 * Hash a pointer using a SplitMix64-style bit-mixing function.
 * This produces a well-distributed 32-bit hash even if the input pointer
 * has low entropy (e.g. aligned addresses).
 */
inline uint32_t hip_hash_ptr(const void* p) 
{
    uintptr_t x = (uintptr_t)p;
    /* mezcla sencilla tipo splitmix */
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

/**
 * Compute the bucket index in the hash map.
 * Assumes HIP_MAP_NUM_BUCKETS is a power of 2.
 * Using '& (N-1)' is faster than modulo and guarantees the index is in range.
 */
inline uint32_t hip_bucket_idx(hipFunction_t key)
{
    return hip_hash_ptr((const void*)key) & (HIP_MAP_NUM_BUCKETS - 1u);
}

/** 
 * Insert a (key → name) entry into the hash map.
 * If the key already exists, update the name.
 * Otherwise, create a new node at the head of the bucket list.
 */
int kernel_map_insert(hipFunction_t function, const char* kernel_name)
{
    uint32_t bucket = hip_bucket_idx(function);
    KernelMapEntry* entry = kernel_map_buckets[bucket];

    // Check if entry already exists (update case)
    for (; entry; entry = entry->next)
    {
        if (entry->function_handle == function)
        {
            // Update stored name
            if (entry->kernel_name)
                free(entry->kernel_name);

            entry->kernel_name = kernel_name ? strdup(kernel_name) : NULL;
            return 1;
        }
    }

    // Entry not found → create new node
    entry = (KernelMapEntry*)malloc(sizeof(KernelMapEntry));
    if (!entry)
        return 0;

    entry->function_handle = function;
    entry->kernel_name = kernel_name ? strdup(kernel_name) : NULL;

    // Insert new entry at head of bucket list
    entry->next = kernel_map_buckets[bucket];
    kernel_map_buckets[bucket] = entry;

    return 1;
}

/**
 * Look up a key in the hash map.
 * Returns 1 if the key exists, 0 otherwise.
 */
int kernel_map_contains(hipFunction_t function)
{
    uint32_t bucket = hip_bucket_idx(function);
    KernelMapEntry* entry = kernel_map_buckets[bucket];
    // Traverse the linked list in this bucket
    for (; entry; entry = entry->next)
    {
        if (entry->function_handle == function)
        {
            return 1;   // Found
        }
    }
    return 0;   // Not found
}

/**
 * Free all entries in the hash map and reset all buckets.
 * This iterates over every bucket, walks each linked list, and frees nodes.
 */
void kernel_map_clear(void)
{
    for (uint32_t bucket = 0; bucket < HIP_MAP_NUM_BUCKETS; ++bucket)
    {
        KernelMapEntry* entry = kernel_map_buckets[bucket];

        // Traverse and free each entry in this bucket
        while (entry)
        {
            KernelMapEntry* next = entry->next;

            if (entry->kernel_name)
                free(entry->kernel_name);

            free(entry);

            entry = next;
        }

        // Reset bucket head
        kernel_map_buckets[bucket] = NULL;
    }
}

int kernel_map_put_safe(hipFunction_t function, const char* name)
{
    int exists;

	//check if already exists
    pthread_rwlock_rdlock(&kernel_map_rwlock);
    exists = kernel_map_contains(function);
    pthread_rwlock_unlock(&kernel_map_rwlock);

    if (exists) return 0;  // already exists

	//recheck if already exists (because another thread could have inserted it meanwhile)
    pthread_rwlock_wrlock(&kernel_map_rwlock);
    exists = kernel_map_contains(function);
    if (exists) {
        pthread_rwlock_unlock(&kernel_map_rwlock);
        return 0; // already exists (2nd check) 
    }

	// insert new entry
    int ok = kernel_map_insert(function, name); // ok = 1 if success, 0 if malloc failed
    pthread_rwlock_unlock(&kernel_map_rwlock);
    return ok ? 1 : -1;   // success or malloc failed
}

int kernel_map_get_safe(hipFunction_t function)
{
	int r;
    pthread_rwlock_rdlock(&kernel_map_rwlock);
    r = kernel_map_contains(function);
    pthread_rwlock_unlock(&kernel_map_rwlock);
    return r;
}

void kernel_map_clear_all_safe(void)
{
    pthread_rwlock_wrlock(&kernel_map_rwlock);
    kernel_map_clear();
    pthread_rwlock_unlock(&kernel_map_rwlock);
}

/****************************************************************************/
/* HIP INSTRUMENTATION                                                     */
/****************************************************************************/

__thread int _hipLaunch_stream = 0;

void Extrae_hipConfigureCall_Enter (dim3 p1, dim3 p2, size_t p3, hipStream_t p4)
{
	int strid;
	int devid;
	unsigned tag = Extrae_HIP_tag_generator();

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);
	UNREFERENCED_PARAMETER(p3);

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_ConfigureCall_Entry ();

	TRACE_USER_COMMUNICATION_EVENT (LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	strid = Extrae_HIP_SearchStream (devid, p4);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in hipConfigureCall (p4=%p)\n", p4);
		exit (-1);
	}
	_hipLaunch_stream = strid;

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPCONFIGKERNEL_GPU_VAL, EVT_BEGIN, tag, tag, 0, 0);
}

void Extrae_hipConfigureCall_Exit (void)
{
	int devid;

	hipGetDevice (&devid);

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, _hipLaunch_stream, HIPCONFIGKERNEL_GPU_VAL, EVT_END, Extrae_HIP_tag_get(), 0, 0, 0);

	Probe_Hip_ConfigureCall_Exit ();
}

void Extrae_hipLaunchKernel_Enter(const char* address, unsigned int gridDim, unsigned int blockDim, size_t sharedMemBytes, hipStream_t stream)
{
	int devid;
	unsigned tag = Extrae_HIP_tag_generator();

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Launch_Entry((UINT64) address);

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	int strid = Extrae_HIP_SearchStream(devid, stream);
	_hipLaunch_stream = strid;

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, _hipLaunch_stream, HIPKERNEL_GPU_VAL, (UINT64)address, tag, sharedMemBytes, gridDim, blockDim); 

}

void Extrae_hipLaunchKernel_Exit(void)
{
	int devid;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, _hipLaunch_stream, HIPKERNEL_GPU_VAL, EVT_END, Extrae_HIP_tag_get(), 0, 0, 0);

	Probe_Hip_Launch_Exit ();
}

void Extrae_hipModuleLaunchKernel_Enter(hipFunction_t f, unsigned gridDim, unsigned int blockDim, size_t sharedMemBytes, hipStream_t stream)
{
	int devid;
	unsigned tag = Extrae_HIP_tag_generator();

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_Launch_Entry((UINT64) f);

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	int strid = Extrae_HIP_SearchStream(devid, stream);
	_hipLaunch_stream = strid;

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, _hipLaunch_stream, HIPKERNEL_GPU_VAL, (UINT64)f, tag, sharedMemBytes, gridDim, blockDim); 

}

void Extrae_hipModuleLaunchKernel_Exit(void)
{
	int devid;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, _hipLaunch_stream, HIPKERNEL_GPU_VAL, EVT_END, Extrae_HIP_tag_get(), 0, 0, 0);

	Probe_Hip_Launch_Exit ();
}

void Extrae_hipMalloc_Enter(unsigned int event, void **devPtr, size_t size)
{
	int devid;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Malloc_Entry(event, (UINT64)devPtr, size);
}

void Extrae_hipMalloc_Exit(unsigned int event)
{
	int devid;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Malloc_Exit(event);
}

void Extrae_hipFree_Enter(unsigned int event, void *devPtr)
{
	int devid;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Free_Entry(event, (UINT64)devPtr);
}

void Extrae_hipFree_Exit(unsigned int event)
{
	int devid;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Free_Exit(event);
}

void Extrae_hipHostAlloc_Enter(void **pHost, size_t size)
{
	Probe_Hip_HostAlloc_Entry((UINT64)pHost, size);
}

void Extrae_hipHostAlloc_Exit(void)
{
	Probe_Hip_HostAlloc_Exit();
}

void Extrae_hipDeviceSynchronize_Enter(void)
{
	int devid;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_ThreadBarrier_Entry ();
	Extrae_HIP_flush_streams(devid, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_hipDeviceSynchronize_Exit(void)
{
	int devid;

	hipGetDevice (&devid);

	Probe_Hip_ThreadBarrier_Exit ();
}

void Extrae_hipThreadSynchronize_Enter(void)
{
	int devid;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_ThreadBarrier_Entry ();
	Extrae_HIP_flush_streams(devid, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_hipThreadSynchronize_Exit(void)
{
	int devid;

	hipGetDevice (&devid);

	Probe_Hip_ThreadBarrier_Exit ();
}

void Extrae_HIP_flush_streams(int device_id, int stream_id)
{
	int d = 0, s = 0;

	if ( devices == NULL )
	{
		return;
	}

	for ( d = (device_id == XTR_FLUSH_ALL_DEVICES ? 0: device_id);
			  d < (device_id == XTR_FLUSH_ALL_DEVICES ? HIPdevices: device_id+1);
			  ++d )
	{
		if ( devices[d].initialized )
		{
			for ( s = (stream_id == XTR_FLUSH_ALL_STREAMS ? 0: stream_id);
						s < (stream_id == XTR_FLUSH_ALL_STREAMS ?  devices[d].nstreams: stream_id+1);
						++s )
			{
				flushGPUEvents (d, s);
			}
		}
	}
}

void Extrae_hipStreamCreate_Enter(hipStream_t *p1)
{
	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.csc.stream = p1;

	Probe_Hip_StreamCreate_Entry();
}

void Extrae_hipStreamCreate_Exit(void)
{
	int devid;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice (&devid);
	Extrae_HIP_Initialize(devid);

	Extrae_HIP_RegisterStream(devid,
	  *Extrae_HIP_saved_params[THREADID].punion.csc.stream);

	Probe_Hip_StreamCreate_Exit();
}

void Extrae_hipStreamDestroy_Enter(hipStream_t stream)
{
	int devid;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Probe_Hip_StreamDestroy_Entry();

	Extrae_HIP_saved_params[THREADID].punion.css.stream = stream;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Extrae_HIP_unRegisterStream(devid, Extrae_HIP_saved_params[THREADID].punion.css.stream);
}

void Extrae_hipStreamDestroy_Exit(void)
{
	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Probe_Hip_StreamDestroy_Exit();
}

void Extrae_hipStreamSynchronize_Enter(hipStream_t p1)
{
	int strid;
	int devid;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.css.stream = p1;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	strid = Extrae_HIP_SearchStream (devid,
	  Extrae_HIP_saved_params[THREADID].punion.css.stream);

	Probe_Hip_StreamBarrier_Entry (devices[devid].Stream[strid].threadid);

	Extrae_HIP_flush_streams(devid, strid);

	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in hipStreamSynchronize\n");
		exit (-1);
	}
}

void Extrae_hipStreamSynchronize_Exit(void)
{
	int strid;
	int devid;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	strid = Extrae_HIP_SearchStream (devid,
	  Extrae_HIP_saved_params[THREADID].punion.css.stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in hipStreamSynchronize\n");
		exit (-1);
	}

	Probe_Hip_StreamBarrier_Exit ();
}

void Extrae_hipMemcpy_Enter(void* p1, const void* p2, size_t p3, enum hipMemcpyKind p4)
{
	int devid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.cm.size = p3;
	Extrae_HIP_saved_params[THREADID].punion.cm.kind = p4;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_Memcpy_Entry (p3);

	tag = Extrae_HIP_tag_generator();

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (p4 == hipMemcpyHostToDevice)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, p3, tag, tag);
	}

	/* If the memcpy is started at host, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Exit
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication starts at this point.
	 */
	if (p4 == hipMemcpyHostToDevice)
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0, HIPMEMCPY_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);
	}
	else if (p4 == hipMemcpyDeviceToHost)
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0, HIPMEMCPY_GPU_VAL, EVT_BEGIN, tag, p3, 0, 0);
	}
}

void Extrae_hipMemcpy_Exit(void)
{
	int devid;
	unsigned tag;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	tag = Extrae_HIP_tag_get();

	/* THIS IS SYMMETRIC TO Extrae_hipMemcpy_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Enter
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication arrives at this point.
	 */
	if (Extrae_HIP_saved_params[THREADID].punion.cm.kind == hipMemcpyHostToDevice ||
	  Extrae_HIP_saved_params[THREADID].punion.cm.kind == hipMemcpyDeviceToDevice)
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0, HIPMEMCPY_GPU_VAL, EVT_END, tag, Extrae_HIP_saved_params[THREADID].punion.cm.size, 0, 0);
	}
	else if (Extrae_HIP_saved_params[THREADID].punion.cm.kind == hipMemcpyDeviceToHost)
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0, HIPMEMCPY_GPU_VAL, EVT_END, 0, Extrae_HIP_saved_params[THREADID].punion.cm.size, 0, 0);
	}

	Probe_Hip_Memcpy_Exit ();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (Extrae_HIP_saved_params[THREADID].punion.cm.kind == hipMemcpyDeviceToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, Extrae_HIP_saved_params[THREADID].punion.cm.size, tag, tag);
	}
}

void Extrae_hipMemcpyAsync_Enter(void* p1, const void* p2, size_t p3, enum hipMemcpyKind p4, hipStream_t p5)
{
	int devid;
	int strid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.cma.size = p3;
	Extrae_HIP_saved_params[THREADID].punion.cma.kind = p4;
	Extrae_HIP_saved_params[THREADID].punion.cma.stream = p5;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_MemcpyAsync_Entry (p3);

	tag = Extrae_HIP_tag_generator();

	/* Emit communication from the host side if memcpykind refers to host to {host,device} */
	if (p4 == hipMemcpyHostToDevice)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV,
		  TASKID, p3, tag, tag);
	}

	strid = Extrae_HIP_SearchStream (devid, p5);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_hipMemcpyAsync_Enter\n");
		exit (-1);
	}

	/* If the memcpy is started at host, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Exit
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication starts at this point.
	 */
	if (p4 == hipMemcpyHostToDevice)
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);
	}
	else if (p4 == hipMemcpyDeviceToHost)
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_BEGIN, tag, p3, 0, 0);
	}
}

void Extrae_hipMemcpyAsync_Exit(void)
{
	int devid;
	int strid;
	unsigned tag;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	tag = Extrae_HIP_tag_get();

	strid = Extrae_HIP_SearchStream (devid,
	  Extrae_HIP_saved_params[THREADID].punion.cma.stream);
	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_hipMemcpyAsync_Exit\n");
		exit (-1);
	}

	/* THIS IS SYMMETRIC TO Extrae_hipMemcpyAsync_Enter */
	/* If the memcpy is started at device, we use tag = 0 to indicate that we
	 * don't want a communication at this point (this will occur at _Enter
	 * point).
	 * If the memcpy was started at the accelerator, we pass a tag != 0 to
	 * indicate that the communication arrives at this point.
	 */
	if (Extrae_HIP_saved_params[THREADID].punion.cma.kind == hipMemcpyHostToDevice ||
	   Extrae_HIP_saved_params[THREADID].punion.cma.kind == hipMemcpyDeviceToDevice)
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_END, tag, Extrae_HIP_saved_params[THREADID].punion.cma.size, 0, 0);
	}
	else
	{
		Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_END, 0, Extrae_HIP_saved_params[THREADID].punion.cma.size, 0, 0);
	}

	Probe_Hip_MemcpyAsync_Exit ();

	/* Emit communication to the host side if memcpykind refers to {host,device} to host */
	if (Extrae_HIP_saved_params[THREADID].punion.cma.kind == hipMemcpyDeviceToHost)
	{
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV,
		  TASKID, Extrae_HIP_saved_params[THREADID].punion.cma.size, tag, tag);
	}
}

void Extrae_hipMemcpyHtoD_Enter(void* p1, const void* p2, size_t p3)
{
	int devid;
	int strid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.cma.size = p3;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Memcpy_Entry(p3);

	tag = Extrae_HIP_tag_generator();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, p3, tag, tag);

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0, HIPMEMCPY_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);

}

void Extrae_hipMemcpyHtoD_Exit(void)
{
	int devid;
	unsigned tag;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	tag = Extrae_HIP_tag_get();

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0,HIPMEMCPY_GPU_VAL, EVT_END, tag, Extrae_HIP_saved_params[THREADID].punion.cm.size, 0, 0);

	Probe_Hip_Memcpy_Exit();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID, Extrae_HIP_saved_params[THREADID].punion.cm.size, tag, tag);

}

void Extrae_hipMemcpyHtoDAsync_Enter(void* p1, const void* p2, size_t p3, hipStream_t p4)
{
	int devid;
	int strid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.cma.size = p3;
	Extrae_HIP_saved_params[THREADID].punion.cma.stream = p4;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_Memcpy_Entry (p3);

	tag = Extrae_HIP_tag_generator();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, p3, tag, tag);

	strid = Extrae_HIP_SearchStream(devid, p4);

	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_hipMemcpyHtoDAsync_Enter\n");
		exit (-1);
	}

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);

}

void Extrae_hipMemcpyHtoDAsync_Exit(void)
{
	int devid;
	unsigned tag;
	int strid;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	tag = Extrae_HIP_tag_get();

	strid = Extrae_HIP_SearchStream(devid, Extrae_HIP_saved_params[THREADID].punion.cma.stream);

	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_hipMemcpyHtoDAsync_Exit\n");
		exit (-1);
	}

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_END, tag, Extrae_HIP_saved_params[THREADID].punion.cma.size, 0, 0);

	Probe_Hip_MemcpyAsync_Exit ();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID, Extrae_HIP_saved_params[THREADID].punion.cma.size, tag, tag);

}

void Extrae_hipMemcpyDtoH_Enter(void* p1, const void* p2, size_t p3)
{
	int devid;
	int strid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.cma.size = p3;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_Memcpy_Entry (p3);

	tag = Extrae_HIP_tag_generator();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, p3, tag, tag);

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0, HIPMEMCPY_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);

}

void Extrae_hipMemcpyDtoH_Exit(void)
{
	int devid;
	unsigned tag;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	tag = Extrae_HIP_tag_get();

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, 0,HIPMEMCPY_GPU_VAL, EVT_END, tag, Extrae_HIP_saved_params[THREADID].punion.cm.size, 0, 0);

	Probe_Hip_Memcpy_Exit();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID, Extrae_HIP_saved_params[THREADID].punion.cm.size, tag, tag);
}

void Extrae_hipMemcpyDtoHAsync_Enter(void* p1, const void* p2, size_t p3, hipStream_t p4)
{
	int devid;
	int strid;
	unsigned tag;

	UNREFERENCED_PARAMETER(p1);
	UNREFERENCED_PARAMETER(p2);

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	Extrae_HIP_saved_params[THREADID].punion.cma.size = p3;
	Extrae_HIP_saved_params[THREADID].punion.cma.stream = p4;

	hipGetDevice (&devid);
	Extrae_HIP_Initialize (devid);

	Probe_Hip_Memcpy_Entry (p3);

	tag = Extrae_HIP_tag_generator();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, p3, tag, tag);

	strid = Extrae_HIP_SearchStream(devid, p4);

	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_hipMemcpyDtoHAsync_Enter\n");
		exit (-1);
	}

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_BEGIN, 0, p3, 0, 0);

}

void Extrae_hipMemcpyDtoHAsync_Exit(void)
{
	int devid;
	unsigned tag;
	int strid;

	ASSERT(Extrae_HIP_saved_params!=NULL, "Unallocated Extrae_HIP_saved_params");

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	tag = Extrae_HIP_tag_get();

	strid = Extrae_HIP_SearchStream(devid, Extrae_HIP_saved_params[THREADID].punion.cma.stream);

	if (strid == -1)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot determine stream index in Extrae_hipMemcpyHtoDAsync_Exit\n");
		exit (-1);
	}

	Extrae_HIP_AddEventToStream(EXTRAE_HIP_NEW_TIME, devid, strid, HIPMEMCPYASYNC_GPU_VAL, EVT_END, tag, Extrae_HIP_saved_params[THREADID].punion.cma.size, 0, 0);

	Probe_Hip_MemcpyAsync_Exit ();

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID, Extrae_HIP_saved_params[THREADID].punion.cma.size, tag, tag);

}

void Extrae_hipMemset_Enter(void *devPtr, size_t count)
{
	int devid;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Memset_Entry((UINT64)devPtr, count);
}

void Extrae_hipMemset_Exit(void)
{
	int devid;

	hipGetDevice(&devid);
	Extrae_HIP_Initialize(devid);

	Probe_Hip_Memset_Exit();
}

void Extrae_reallocate_HIP_info(unsigned old_threads, unsigned nthreads)
{
	Extrae_HIP_saved_params = (Extrae_hip_saved_params_t*) xrealloc (
		Extrae_HIP_saved_params, sizeof(Extrae_hip_saved_params_t)*nthreads);

	memset(&Extrae_HIP_saved_params[old_threads], 0, sizeof(Extrae_hip_saved_params_t)*(nthreads-old_threads));

	if (Extrae_HIP_saved_params == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Error! Cannot reallocate HIP parameters buffers per thread!\n");
		exit (-1);
	}
}

void Extrae_hipDeviceReset_Enter(void)
{
	Probe_Hip_DeviceReset_Enter();
	Extrae_HIP_flush_streams(XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);
}

void Extrae_hipDeviceReset_Exit(void)
{
	int devid;
	hipGetDevice(&devid);

	Extrae_HIP_deInitialize(devid);
	Probe_Hip_DeviceReset_Exit();
}

void Extrae_hipThreadExit_Enter (void)
{
	Probe_Hip_ThreadExit_Enter();
}

void Extrae_hipThreadExit_Exit (void)
{
	int devid;
	hipGetDevice (&devid);

	Extrae_HIP_deInitialize (devid);
	Probe_Hip_ThreadExit_Exit();
}

void Extrae_hipEventRecord_Enter(void)
{
	Probe_Hip_EventRecord_Entry();
}

void Extrae_hipEventRecord_Exit(void)
{
	Probe_Hip_EventRecord_Exit();
}

void Extrae_hipEventSynchronize_Enter(void)
{
	Probe_Hip_EventSynchronize_Entry();
}

void Extrae_hipEventSynchronize_Exit(void)
{
	Probe_Hip_EventSynchronize_Exit();
}

/**
 * Performing the flush of streams events during the finalization of Extrae in the destructor call
 * triggers HIP calls after the HIP library has been unloaded resulting in crashes.
 * To solve that case we always call Extrae_HIP_finalize with atexit and mark hipInitialized to 0
 * to prevent entering this routine after the return point of the main.
 */
void Extrae_HIP_finalize(void)
{
	Backend_Enter_Instrumentation();
	if (EXTRAE_INITIALIZED() && hipInitialized == 1)
	{
		for (int i = 0; i < HIP_MAP_NUM_BUCKETS; ++i)
		{
			KernelMapEntry* n = kernel_map_buckets[i];
			while(n)
			{
				KernelMapEntry* nxt = n->next;
				int lineno = 0;
				Extrae_AddFunctionDefinitionEntryToLocalSYM('Y', (UINT64)n->function_handle, n->kernel_name, "??", lineno);
				n = nxt;
			}
		}
		
		Extrae_HIP_flush_streams(XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);
		
		for(int i = 0; i <HIPdevices ; ++i){
			if(devices[i].initialized == TRUE){
				gpuEventList_free(&devices[i].availableEvents);
				Extrae_HIP_deInitialize (i);
			}
		}
		xfree(devices);

		devices = NULL;
		HIPdevices = 0;

		hipInitialized = 0;
	}
	Backend_Leave_Instrumentation();
}