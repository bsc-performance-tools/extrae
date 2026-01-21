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

#define TRACE_GPU_EVENT(_thread, _evttime, _evttype, _evtvalue, _evtbegin, _evtsize) \
{                                                                              \
    int _thread_id = _thread;                                                  \
    event_t _evt;                                                              \
    if (tracejant && TracingBitmap[TASKID])                                    \
    {                                                                          \
        _evt.time = _evttime;                                                  \
        _evt.event = _evttype;                                                 \
        _evt.value = _evtvalue;                                                \
        _evt.param.gpu_param.begin = _evtbegin;                                \
        _evt.param.gpu_param.memSize = _evtsize;                               \
        HARDWARE_COUNTERS_READ(_thread_id, _evt, FALSE);                       \
        BUFFER_INSERT(_thread_id, TRACING_BUFFER(_thread_id), _evt);           \
    }                                                                          \
}

#define TRACE_GPU_KERNEL_EVENT(_thread, _evttime, _evttype, _evtvalue, _blockspergrid, _threadsperblock) \
{                                                                              \
    event_t _evt;                                                              \
    if (tracejant && TracingBitmap[TASKID])                                    \
    {                                                                          \
        _evt.time = _evttime;                                                  \
        _evt.event = _evttype;                                                 \
        _evt.value = _evtvalue;                                                \
        _evt.param.gpu_param.gridSize = _blockspergrid;                        \
        _evt.param.gpu_param.blockSize = _threadsperblock;                     \
        HARDWARE_COUNTERS_READ(_thread, _evt, FALSE);                          \
        BUFFER_INSERT(_thread, TRACING_BUFFER(_thread), _evt);                 \
    }                                                                          \
}

#endif /* TRACE_MACROS_GPU_H_INCLUDED */
