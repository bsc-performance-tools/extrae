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

#ifndef __OPENSHMEM_TRACE_MACROS_H__
#define __OPENSHMEM_TRACE_MACROS_H__

#include "trace_macros.h"

#if USE_HARDWARE_COUNTERS

#define TRACE_OPENSHMEM_CALLERS(evttime, evtvalue)           \
{                                                            \
  if ( evtvalue == EVT_BEGIN )                               \
  {                                                          \
    Extrae_trace_callers (evttime, FOUR_CALLS_AGO, CALLER_MPI); \
  }                                                          \
}

#define TRACE_OPENSHMEM_EVENT_AND_COUNTERS(evttime,evttype,evtvalue,evtsize) \
{                                                                            \
  int thread_id = THREADID;                                                  \
  event_t evt;                                                               \
  if (tracejant)                                                             \
  {                                                                          \
    evt.time = (evttime);                                                    \
    evt.event = (evttype);                                                   \
    evt.value = (evtvalue);                                                  \
    evt.param.mpi_param.size = (evtsize);                                    \
    HARDWARE_COUNTERS_READ(thread_id, evt, 1);                               \
    BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);                \
    TRACE_OPENSHMEM_CALLERS(evttime, evtvalue);                              \
  }                                                                          \
}

#else

#define TRACE_OPENSHMEM_EVENT_AND_COUNTERS(evttime,evttype,evtvalue,evtsize) \
{                                                                            \
  TRACE_OPENSHMEM_EVENT(evttime,evttype,evtvalue,evtsize);                   \
}

#endif

#define TRACE_OPENSHMEM_EVENT(evttime,evttype,evtvalue,evtsize)              \
{                                                                            \
  int thread_id = THREADID;                                                  \
  event_t evt;                                                               \
  if (tracejant)                                                             \
  {                                                                          \
    evt.time = (evttime);                                                    \
    evt.event = (evttype);                                                   \
    evt.value = (evtvalue);                                                  \
    evt.param.mpi_param.size = (evtsize);                                    \
    HARDWARE_COUNTERS_READ(thread_id, evt, 0);                               \
    BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);                \
    TRACE_OPENSHMEM_CALLERS(evttime, evtvalue);                              \
  }                                                                          \
}

#endif /* __OPENSHMEM_TRACE_MACROS_H__ */
