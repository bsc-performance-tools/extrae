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

#ifndef HIP_PROBE_H_INCLUDED
#define HIP_PROBE_H_INCLUDED

void Probe_Hip_ConfigureCall_Entry(void);
void Probe_Hip_ConfigureCall_Exit(void);
void Probe_Hip_Launch_Entry(UINT64 p1);
void Probe_Hip_Launch_Exit(void);
void Probe_Hip_Malloc_Entry(unsigned int, UINT64, size_t);
void Probe_Hip_Malloc_Exit(unsigned int);
void Probe_Hip_Free_Entry(unsigned int, UINT64);
void Probe_Hip_Free_Exit(unsigned int);
void Probe_Hip_HostAlloc_Entry(UINT64, size_t);
void Probe_Hip_HostAlloc_Exit();
void Probe_Hip_Memcpy_Entry(size_t size);
void Probe_Hip_Memcpy_Exit(void);
void Probe_Hip_MemcpyAsync_Entry(size_t size);
void Probe_Hip_MemcpyAsync_Exit (void);
void Probe_Hip_Memset_Entry(UINT64, size_t);
void Probe_Hip_Memset_Exit();
void Probe_Hip_ThreadBarrier_Entry(void);
void Probe_Hip_ThreadBarrier_Exit(void);
void Probe_Hip_StreamBarrier_Entry(unsigned thread);
void Probe_Hip_StreamBarrier_Exit(void);
void Probe_Hip_DeviceReset_Enter(void);
void Probe_Hip_DeviceReset_Exit(void);
void Probe_Hip_ThreadExit_Enter(void);
void Probe_Hip_ThreadExit_Exit(void);
void Probe_Hip_StreamCreate_Entry(void);
void Probe_Hip_StreamCreate_Exit(void);
void Probe_Hip_StreamDestroy_Entry(void);
void Probe_Hip_StreamDestroy_Exit(void);

void Probe_Hip_EventRecord_Entry(void);
void Probe_Hip_EventRecord_Exit(void);

void Probe_Hip_EventSynchronize_Entry(void);
void Probe_Hip_EventSynchronize_Exit(void);

void Extrae_set_trace_HIP(int b);
int Extrae_get_trace_HIP(void);

#endif
