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

#include <stdio.h>
#include "common.h"
#include "threadid.h"
#include "wrapper.h"
#include "gpu_common.h"
#include "gpu_probe.h"
#include "gpu_macros.h" /* GPU() */

int trace_gpu = TRUE;

void Extrae_set_trace_GPU(int b) { trace_gpu = b; }
int Extrae_get_trace_GPU(void) { return trace_gpu; }
static struct DeviceInfo_t *deviceArray = NULL;

#if 0
#define DEBUG fprintf(stdout, "THREAD %d: %s\n", THREADID, __FUNCTION__)
#else
#define DEBUG
#endif

#define GPU_PROBE_ACTIVE() (mpitrace_on && Extrae_get_trace_GPU())

void Probe_Gpu_ConfigureCall_Entry(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(CONFIGCALL_VAL), EVT_BEGIN);
}

void Probe_Gpu_ConfigureCall_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(CONFIGCALL_VAL), EVT_END);
}

void Probe_Gpu_Launch_Entry(GPU_FUNCTION_T p1, unsigned int blocksPerGrid, unsigned int threadsPerBlock, size_t sharedMemBytes, GPU_STREAM_T stream, GPU_CONTEXT_T ctx)
{

	int device_id = -1;
	int stream_idx = -1;

	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(LAUNCH_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_KERNEL_INST_EV), (UINT64)p1);
	}

	unsigned tag = GetGPUCommTag();
	GPU_thread_args.tag = tag;

	GPU_GET_DEVICE_SAFE(ctx, device_id);
	stream_idx = RegisterStream(device_id, stream);
	GPU_thread_args.stream_idx = stream_idx;

	TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, 0, tag, tag);

	AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, GPUEV(KERNEL_GPU_VAL), (UINT64)p1, tag, sharedMemBytes, blocksPerGrid, threadsPerBlock);

}

void Probe_Gpu_Launch_Exit(GPU_CONTEXT_T ctx)
{
	int device_id = -1;
	GPU_GET_DEVICE_SAFE(ctx, device_id);
	unsigned tag = GPU_thread_args.tag;

	AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, GPU_thread_args.stream_idx, GPUEV(KERNEL_GPU_VAL), EVT_END, tag, 0, 0, 0);

	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(LAUNCH_VAL), EVT_END);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_KERNEL_INST_EV), EVT_END);
	}
}

void Probe_Gpu_Malloc_Entry(unsigned int event, UINT64 ptr, size_t size)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), event, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_PTR_EV), ptr);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), size);
	}
}

void Probe_Gpu_Malloc_Exit(unsigned int event)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), event, EVT_END);
}

void Probe_Gpu_Free_Entry(unsigned int event, UINT64 devPtr)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), event, EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_PTR_EV), devPtr);
	}
}

void Probe_Gpu_Free_Exit(unsigned int event)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), event, EVT_END);
}

void Probe_Gpu_HostAlloc_Entry(UINT64 ptr, size_t size)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(HOSTALLOC_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_PTR_EV), ptr);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), size);
	}
}

void Probe_Gpu_HostAlloc_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(HOSTALLOC_VAL), EVT_END);
}

void Probe_Gpu_Memcpy_Entry(size_t size)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(MEMCPY_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), size);
	}
}

void Probe_Gpu_Memcpy_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(MEMCPY_VAL), EVT_END);
}

void Probe_Gpu_MemcpyToSymbol_Entry(size_t size)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(MEMCPYTOSYMBOL_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), size);
	}
}

void Probe_Gpu_MemcpyToSymbol_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(MEMCPYTOSYMBOL_VAL), EVT_END);
}

void Probe_Gpu_MemcpyFromSymbol_Entry(size_t size)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(MEMCPYFROMSYMBOL_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), size);
	}
}

void Probe_Gpu_MemcpyFromSymbol_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(MEMCPYFROMSYMBOL_VAL), EVT_END);
}

void Probe_Gpu_MemcpyAsync_Entry(void *dst, const void *src, size_t size, GPU_MEMCPY_KIND_T kind, GPU_STREAM_T stream, GPU_CONTEXT_T ctx)
{

	UNREFERENCED_PARAMETER(dst);
	UNREFERENCED_PARAMETER(src);

	int device_id = -1;
	int stream_idx = -1;

	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(MEMCPYASYNC_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), size);
	}

	unsigned tag = GetGPUCommTag();

	GPU_thread_args.stream = stream;
	GPU_thread_args.memcpyKind = kind;
	GPU_thread_args.memcpySize = size;
	GPU_thread_args.tag = tag;

	GPU_GET_DEVICE_SAFE(ctx, device_id);

	stream_idx = RegisterStream(device_id, stream);
	GPU_thread_args.stream_idx = stream_idx;

	if (kind == GPU_MEMCPY_H2D_T)
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, size, tag, tag);

	if (kind == GPU_MEMCPY_D2H_T)
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, GPUEV(MEMCPYASYNC_GPU_VAL), EVT_BEGIN, tag, size, 0, 0);
	else
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, GPUEV(MEMCPYASYNC_GPU_VAL), EVT_BEGIN, 0, size, 0, 0);

}

void Probe_Gpu_MemcpyAsync_Exit(GPU_CONTEXT_T ctx)
{

	int device_id = -1;
	int stream_idx = GPU_thread_args.stream_idx;
	GPU_MEMCPY_KIND_T kind = GPU_thread_args.memcpyKind;
	size_t size = GPU_thread_args.memcpySize;
	unsigned tag = GPU_thread_args.tag;

	GPU_GET_DEVICE_SAFE(ctx, device_id);

	if (kind == GPU_MEMCPY_H2D_T)
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, GPUEV(MEMCPYASYNC_GPU_VAL), EVT_END, tag, size, 0, 0);
	else
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, GPUEV(MEMCPYASYNC_GPU_VAL), EVT_END, 0, size, 0, 0);

	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(MEMCPYASYNC_VAL), EVT_END);

	if (kind == GPU_MEMCPY_D2H_T)
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID, size, tag, tag);

}

void Probe_Gpu_Memset_Entry(UINT64 devPtr, size_t count)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(MEMSET_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_PTR_EV), devPtr);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), count);
	}
}

void Probe_Gpu_Memset_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(MEMSET_VAL), EVT_END);
}

void Probe_Gpu_MemsetAsync_Entry(UINT64 devPtr, size_t count)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(MEMSETASYNC_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_PTR_EV), devPtr);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_DYNAMIC_MEM_SIZE_EV), count);
	}
}

void Probe_Gpu_MemsetAsync_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(MEMSETASYNC_VAL), EVT_END);
}

void Probe_Gpu_ThreadBarrier_Entry(GPU_CONTEXT_T ctx)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(THREADBARRIER_VAL), EVT_BEGIN);

	int device_id = -1;
	GPU_GET_DEVICE_SAFE(ctx, device_id);
	FlushStreams(device_id, XTR_FLUSH_ALL_STREAMS);
}

void Probe_Gpu_ThreadBarrier_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(THREADBARRIER_VAL), EVT_END);
}

void Probe_Gpu_StreamBarrier_Entry(GPU_STREAM_T stream, GPU_CONTEXT_T ctx)
{
	int device_id = -1;
	int stream_idx = -1;
	if (GPU_PROBE_ACTIVE()) {
		GPU_GET_DEVICE_SAFE(ctx, device_id);
		stream_idx = RegisterStream(device_id, stream);
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(STREAMSYNCHRONIZE_VAL), EVT_BEGIN);
		int threadid = deviceArray[device_id].streams[stream_idx].thread_id;
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_STREAM_DEST_ID_EV), threadid + 1);
		FlushStreams(device_id, stream_idx);
	}
}

void Probe_Gpu_StreamBarrier_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(STREAMSYNCHRONIZE_VAL), EVT_END);
}

void Probe_Gpu_DeviceReset_Entry(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(DEVICERESET_VAL), EVT_BEGIN);
	FlushStreams(XTR_FLUSH_ALL_DEVICES, XTR_FLUSH_ALL_STREAMS);
}

void Probe_Gpu_DeviceReset_Exit(GPU_CONTEXT_T ctx)
{
	int device_id = -1;
	GPU_GET_DEVICE_SAFE(ctx, device_id);
	DeinitializeDevice(device_id);
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(DEVICERESET_VAL), EVT_END);
}

void Probe_Gpu_ThreadExit_Entry(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(THREADEXIT_VAL), EVT_BEGIN);
}

void Probe_Gpu_ThreadExit_Exit(GPU_CONTEXT_T ctx)
{
	int device_id = -1;
	GPU_GET_DEVICE_SAFE(ctx, device_id);
	DeinitializeDevice(device_id);
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(THREADEXIT_VAL), EVT_END);
}

void Probe_Gpu_StreamCreate_Entry(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(STREAMCREATE_VAL), EVT_BEGIN);
}

void Probe_Gpu_StreamCreate_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(STREAMCREATE_VAL), EVT_END);
}

void Probe_Gpu_StreamDestroy_Entry(GPU_STREAM_T stream, GPU_CONTEXT_T ctx)
{
	int device_id = -1;
	GPU_GET_DEVICE_SAFE(ctx, device_id);
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(STREAMDESTROY_VAL), EVT_BEGIN);
	UnregisterStream(device_id, stream);
}

void Probe_Gpu_StreamDestroy_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(STREAMDESTROY_VAL), EVT_END);
}

void Probe_Gpu_StreamRegister_Entry(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(_STREAM_REGISTER_EV), EVT_BEGIN, EMPTY);
}

void Probe_Gpu_StreamRegister_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(_STREAM_REGISTER_EV), EVT_END, EMPTY);
}

void Probe_Gpu_EventRecord_Entry(GPU_EVENT_T event, GPU_STREAM_T stream, GPU_CONTEXT_T ctx)
{
	int device_id = -1;
	int stream_idx = -1;
	GPU_GET_DEVICE_SAFE(ctx, device_id);
	stream_idx = RegisterStream(device_id, stream);
	int threadid = deviceArray[device_id].streams[stream_idx].thread_id;
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(EVENTRECORD_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(EVENT_ID_EV), (UINT64)event);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(_STREAM_DEST_ID_EV), threadid + 1);
	}
}

void Probe_Gpu_EventRecord_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(EVENTRECORD_VAL), EVT_END);
}

void Probe_Gpu_EventSynchronize_Entry(UINT64 GPU_EVENT)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(EVENTSYNCHRONIZE_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(EVENT_ID_EV), GPU_EVENT);
	}
}

void Probe_Gpu_EventSynchronize_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(EVENTSYNCHRONIZE_VAL), EVT_END);
}

void Probe_Gpu_StreamWaitEvent_Entry(UINT64 GPU_EVENT)
{
	if (GPU_PROBE_ACTIVE())
	{
		TRACE_MISCEVENTANDCOUNTERS(LAST_READ_TIME, GPUEV(CALL_EV), GPUEV(STREAMWAITEVENT_VAL), EVT_BEGIN);
		TRACE_EVENT(LAST_READ_TIME, GPUEV(EVENT_ID_EV), GPU_EVENT);
	}
}

void Probe_Gpu_StreamWaitEvent_Exit(void)
{
	if (GPU_PROBE_ACTIVE())
		TRACE_MISCEVENTANDCOUNTERS(TIME, GPUEV(CALL_EV), GPUEV(STREAMWAITEVENT_VAL), EVT_END);
}

/* ── Memcpy helpers ─────────────────────────────────────────────────────── */

void _Probe_gpuMemcpy_Enter(void *dst, const void *src, size_t count, GPU_MEMCPY_KIND_T kind, GPU_CONTEXT_T ctx, void (*entry_probe)(size_t), unsigned long long gpu_value) {
	UNREFERENCED_PARAMETER(dst);
	UNREFERENCED_PARAMETER(src);

	int device_id = -1;
	int stream_idx = -1;
	unsigned tag = GetGPUCommTag();

	GPU_thread_args.memcpyKind = kind;
	GPU_thread_args.memcpySize = count;
	GPU_thread_args.tag = tag;

	GPU_GET_DEVICE_SAFE(ctx, device_id);
	stream_idx = RegisterStream(device_id, GPU_STREAM_DEFAULT);
	GPU_thread_args.stream_idx = stream_idx;

	entry_probe(count);

	if (kind == GPU_MEMCPY_H2D_T)
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_SEND_EV, TASKID, count, tag, tag);

	if (kind == GPU_MEMCPY_D2H_T)
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, gpu_value, EVT_BEGIN, tag, count, 0, 0);
	else
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, gpu_value, EVT_BEGIN, 0, count, 0, 0);
}

void _Probe_gpuMemcpy_Exit(GPU_CONTEXT_T ctx, void (*exit_probe)(void), unsigned long long gpu_value) {
	int device_id = -1;
	int stream_idx = GPU_thread_args.stream_idx;
	GPU_MEMCPY_KIND_T kind = GPU_thread_args.memcpyKind;
	size_t size = GPU_thread_args.memcpySize;
	unsigned tag = GPU_thread_args.tag;

	GPU_GET_DEVICE_SAFE(ctx, device_id);

	if (kind == GPU_MEMCPY_H2D_T)
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, gpu_value, EVT_END, tag, size, 0, 0);
	else
		AddEventToStream(EXTRAE_GPU_NEW_TIME, device_id, stream_idx, gpu_value, EVT_END, 0, size, 0, 0);

	exit_probe();

	if (kind == GPU_MEMCPY_D2H_T)
		TRACE_USER_COMMUNICATION_EVENT(LAST_READ_TIME, USER_RECV_EV, TASKID, size, tag, tag);
}

/* ── Sync memcpy variants ───────────────────────────────────────────────── */

void Probe_gpuMemcpy_Enter(void *dst, const void *src, size_t count, GPU_MEMCPY_KIND_T kind, GPU_CONTEXT_T ctx) {
	_Probe_gpuMemcpy_Enter(dst, src, count, kind, ctx, Probe_Gpu_Memcpy_Entry, GPUEV(MEMCPY_GPU_VAL));
}

void Probe_gpuMemcpy_Exit(GPU_CONTEXT_T ctx) {
	_Probe_gpuMemcpy_Exit(ctx, Probe_Gpu_Memcpy_Exit, GPUEV(MEMCPY_GPU_VAL));
}

void Probe_gpuMemcpyToSymbol_Enter(void *dst, const void *src, size_t count, GPU_MEMCPY_KIND_T kind, GPU_CONTEXT_T ctx) {
	_Probe_gpuMemcpy_Enter(dst, src, count, kind, ctx, Probe_Gpu_MemcpyToSymbol_Entry, GPUEV(MEMCPYTOSYMBOL_GPU_VAL));
}

void Probe_gpuMemcpyToSymbol_Exit(GPU_CONTEXT_T ctx) {
	_Probe_gpuMemcpy_Exit(ctx, Probe_Gpu_MemcpyToSymbol_Exit, GPUEV(MEMCPYTOSYMBOL_GPU_VAL));
}

void Probe_gpuMemcpyFromSymbol_Enter(void *dst, const void *src, size_t count, GPU_MEMCPY_KIND_T kind, GPU_CONTEXT_T ctx) {
	_Probe_gpuMemcpy_Enter(dst, src, count, kind, ctx, Probe_Gpu_MemcpyFromSymbol_Entry, GPUEV(MEMCPYFROMSYMBOL_GPU_VAL));
}

void Probe_gpuMemcpyFromSymbol_Exit(GPU_CONTEXT_T ctx) {
	_Probe_gpuMemcpy_Exit(ctx, Probe_Gpu_MemcpyFromSymbol_Exit, GPUEV(MEMCPYFROMSYMBOL_GPU_VAL));
}
