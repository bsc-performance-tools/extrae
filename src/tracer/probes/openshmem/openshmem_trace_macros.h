#ifndef __OPENSHMEM_TRACE_MACROS_H__
#define __OPENSHMEM_TRACE_MACROS_H__

#include "trace_macros.h"

#if USE_HARDWARE_COUNTERS

#define TRACE_OPENSHMEM_CALLERS(evttime, evtvalue)           \
{                                                            \
  if ( evtvalue == EVT_BEGIN )                               \
  {                                                          \
    trace_callers (evttime, FOUR_CALLS_AGO, CALLER_MPI);     \
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
