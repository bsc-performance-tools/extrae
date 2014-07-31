#ifndef __OPENSHMEM_TRACE_MACROS_H__
#define __OPENSHMEM_TRACE_MACROS_H__

#include "trace_macros.h"

#if USE_HARDWARE_COUNTERS

#define TRACE_OPENSHMEM_EVENT_AND_COUNTERS(evttime,evttype,evtvalue,evtparam) \
{                                                                             \
  int thread_id = THREADID;                                                   \
  event_t evt;                                                                \
  if (tracejant)                                                              \
  {                                                                           \
    evt.time = (evttime);                                                     \
    evt.event = (OPENSHMEM_TYPE);                                             \
    evt.value = (evttype);                                                    \
    evt.param.omp_param.param = (evtvalue);                                   \
    HARDWARE_COUNTERS_READ(thread_id, evt, 1);                                \
    BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);                 \
  }                                                                           \
}

#else

#define TRACE_OPENSHMEM_EVENT_AND_COUNTERS(evttime,evttype,evtvalue,evtparam) \
{                                                                             \
  TRACE_OPENSHMEM_EVENT(evttime,evttype,evtvalue,evtparam);                   \
}

#endif

#define TRACE_OPENSHMEM_EVENT(evttime,evttype,evtvalue,evtparam)              \
{                                                                             \
  int thread_id = THREADID;                                                   \
  event_t evt;                                                                \
  if (tracejant)                                                              \
  {                                                                           \
    evt.time = (evttime);                                                     \
    evt.event = (OPENSHMEM_TYPE);                                             \
    evt.value = (evttype);                                                    \
    evt.param.omp_param.param = (evtvalue);                                   \
    HARDWARE_COUNTERS_READ(thread_id, evt, 0);                                \
    BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);                 \
  }                                                                           \
}

#endif /* __OPENSHMEM_TRACE_MACROS_H__ */
