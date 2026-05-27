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

#ifndef GPU_PROBE_H_INCLUDED
#define GPU_PROBE_H_INCLUDED

void Probe_Gpu_ConfigureCall_Entry(void);
void Probe_Gpu_ConfigureCall_Exit(void);
void Probe_Gpu_Launch_Entry(GPU_FUNCTION_T p1, unsigned int blocksPerGrid, unsigned int threadsPerBlock, size_t sharedMemBytes, GPU_STREAM_T stream, GPU_CONTEXT_T ctx);
void Probe_Gpu_Launch_Exit(GPU_CONTEXT_T ctx);
void Probe_Gpu_Malloc_Entry(unsigned int, UINT64, size_t);
void Probe_Gpu_Malloc_Exit(unsigned int);
void Probe_Gpu_Free_Entry(unsigned int, UINT64);
void Probe_Gpu_Free_Exit(unsigned int);
void Probe_Gpu_HostAlloc_Entry(UINT64, size_t);
void Probe_Gpu_HostAlloc_Exit(void);
void Probe_Gpu_Memcpy_Entry(size_t size);
void Probe_Gpu_Memcpy_Exit(void);
void Probe_Gpu_MemcpyToSymbol_Entry(size_t size);
void Probe_Gpu_MemcpyToSymbol_Exit(void);
void Probe_Gpu_MemcpyFromSymbol_Entry(size_t size);
void Probe_Gpu_MemcpyFromSymbol_Exit(void);
void Probe_Gpu_MemcpyAsync_Entry(void*, const void*, size_t, GPU_MEMCPY_KIND_T, GPU_STREAM_T, GPU_CONTEXT_T);
void Probe_Gpu_MemcpyAsync_Exit(GPU_CONTEXT_T);
void Probe_Gpu_Memset_Entry(UINT64, size_t);
void Probe_Gpu_Memset_Exit(void);
void Probe_Gpu_MemsetAsync_Entry(UINT64, size_t);  
void Probe_Gpu_MemsetAsync_Exit(void);
void Probe_Gpu_ThreadBarrier_Entry(GPU_CONTEXT_T);
void Probe_Gpu_ThreadBarrier_Exit(void);
void Probe_Gpu_StreamBarrier_Entry(GPU_STREAM_T, GPU_CONTEXT_T);
void Probe_Gpu_StreamBarrier_Exit(void);
void Probe_Gpu_DeviceReset_Entry(void);
void Probe_Gpu_DeviceReset_Exit(GPU_CONTEXT_T);
void Probe_Gpu_ThreadExit_Entry(void);
void Probe_Gpu_ThreadExit_Exit(GPU_CONTEXT_T);
void Probe_Gpu_StreamCreate_Entry(void);
void Probe_Gpu_StreamCreate_Exit(void);
void Probe_Gpu_StreamDestroy_Entry(GPU_STREAM_T, GPU_CONTEXT_T);
void Probe_Gpu_StreamDestroy_Exit(void);
void Probe_Gpu_EventRecord_Entry(GPU_EVENT_T, GPU_STREAM_T, GPU_CONTEXT_T);
void Probe_Gpu_EventRecord_Exit(void);
void Probe_Gpu_EventSynchronize_Entry(UINT64);
void Probe_Gpu_EventSynchronize_Exit(void);
void Probe_Gpu_StreamWaitEvent_Entry(UINT64);
void Probe_Gpu_StreamWaitEvent_Exit(void);

void Extrae_set_trace_GPU(int b);
int Extrae_get_trace_GPU(void);

void Probe_Gpu_StreamRegister_Entry(void);
void Probe_Gpu_StreamRegister_Exit(void);

void Probe_gpuMemcpy_Enter(void*, const void*, size_t, GPU_MEMCPY_KIND_T, GPU_CONTEXT_T);
void Probe_gpuMemcpy_Exit(GPU_CONTEXT_T);
void Probe_gpuMemcpyToSymbol_Enter(void*, const void*, size_t, GPU_MEMCPY_KIND_T, GPU_CONTEXT_T);
void Probe_gpuMemcpyToSymbol_Exit(GPU_CONTEXT_T);
void Probe_gpuMemcpyFromSymbol_Enter(void*, const void*, size_t, GPU_MEMCPY_KIND_T, GPU_CONTEXT_T);
void Probe_gpuMemcpyFromSymbol_Exit(GPU_CONTEXT_T);

#endif