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

#ifndef TRACE_MACROS_GPU_H_INCLUDED
#define TRACE_MACROS_GPU_H_INCLUDED

#define TRACE_GPU_EVENT(thread, evttime, evttype, evtvalue, evtbegin, evtsize) \
{                                                                              \
    int thread_id = thread;                                                    \
    event_t evt;                                                               \
    if (tracejant && TracingBitmap[TASKID])                                    \
    {                                                                          \
        evt.time = evttime;                                                    \
        evt.event = evttype;                                                   \
        evt.value = evtvalue;                                                  \
        evt.param.gpu_param.begin = evtbegin;                                  \
        evt.param.gpu_param.memSize = evtsize;                                 \
        HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);                         \
        BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);              \
    }                                                                          \
}

#define TRACE_GPU_KERNEL_EVENT(thread, evttime, evttype, evtvalue, blockspergrid, threadsperblock) \
{                                                                              \
    event_t evt;                                                               \
    if (tracejant && TracingBitmap[TASKID])                                    \
    {                                                                          \
        evt.time = evttime;                                                    \
        evt.event = evttype;                                                   \
        evt.value = evtvalue;                                                  \
        evt.param.gpu_param.gridSize = blockspergrid;                          \
        evt.param.gpu_param.blockSize = threadsperblock;                       \
        HARDWARE_COUNTERS_READ(thread, evt, FALSE);                            \
        BUFFER_INSERT(thread, TRACING_BUFFER(thread), evt);                    \
    }                                                                          \
}

#endif /* TRACE_MACROS_GPU_H_INCLUDED */
