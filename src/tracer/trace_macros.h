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

#ifndef __TRACE_MACROS_H__
#define __TRACE_MACROS_H__

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#include "common.h"
#include "events.h"
#include "clock.h"
#include "calltrace.h" 
#include "mode.h"
#include "trace_buffers.h"
#include "trace_hwc.h"

#if defined(MPI_SUPPORT)
# define TRACING_BITMAP_VALID_EVTYPE(evttype) \
   ((evttype) == MPI_BSEND_EV || (evttype) == MPI_SSEND_EV || (evttype) == MPI_RSEND_EV || \
    (evttype) == MPI_SEND_EV  || (evttype) == MPI_IRECVED_EV || (evttype) == MPI_RECV_EV)
# define TRACING_BITMAP_VALID_EVTARGET(evttarget)                              \
    (((long)evttarget) != MPI_ANY_SOURCE && ((long)evttarget) != MPI_PROC_NULL && ((long)evttarget) != MPI_UNDEFINED) 

#else
# define TRACING_BITMAP_VALID_EVTYPE(evttype)     (TRUE)
# define TRACING_BITMAP_VALID_EVTARGET(evttarget) (TRUE)
# define COMM_STATS_WRAPPER(x)
#endif /* MPI_SUPPORT */

#include "trace_macros_mpi.h"
#include "trace_macros_omp.h"

#define SAMPLE_EVENT_HWC(evttime,evttype,evtvalue)               \
{                                                                \
	event_t evt;                                                   \
	int thread_id = THREADID;                                      \
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)) && TracingBitmap[TASKID])               \
	{                                                              \
		evt.time = (evttime);                                        \
		evt.event = (evttype);                                       \
		evt.value = (evtvalue);                                      \
		/* We don't read counters right now */                       \
		HARDWARE_COUNTERS_READ(thread_id, evt, TRUE);                \
		BUFFER_INSERT(thread_id, SAMPLING_BUFFER(thread_id), evt);   \
	}                                                              \
}

#define SAMPLE_EVENT_NOHWC(evttime,evttype,evtvalue)             \
{                                                                \
	event_t evt;                                                   \
	int thread_id = THREADID;                                      \
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)) && TracingBitmap[TASKID])               \
	{                                                              \
		evt.time = (evttime);                                        \
		evt.event = (evttype);                                       \
		evt.value = (evtvalue);                                      \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);               \
		BUFFER_INSERT(thread_id, SAMPLING_BUFFER(thread_id), evt);   \
	}                                                              \
}

#define SAMPLE_EVENT_HWC_PARAM(evttime,evttype,evtvalue,evtparam)               \
{                                                                \
	event_t evt;                                                   \
	int thread_id = THREADID;                                      \
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)) && TracingBitmap[TASKID])               \
	{                                                              \
		evt.time = (evttime);                                        \
		evt.event = (evttype);                                       \
		evt.value = (evtvalue);                                      \
		evt.param.misc_param.param = (unsigned long long) (evtparam); \
		/* We don't read counters right now */                       \
		HARDWARE_COUNTERS_READ(thread_id, evt, TRUE);                \
		BUFFER_INSERT(thread_id, SAMPLING_BUFFER(thread_id), evt);   \
	}                                                              \
}

#define SAMPLE_EVENT_NOHWC_PARAM(evttime,evttype,evtvalue,evtparam)             \
{                                                                \
	event_t evt;                                                   \
	int thread_id = THREADID;                                      \
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)) && TracingBitmap[TASKID])               \
	{                                                              \
		evt.time = (evttime);                                        \
		evt.event = (evttype);                                       \
		evt.value = (evtvalue);                                      \
		evt.param.misc_param.param = (unsigned long long) (evtparam); \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);               \
		BUFFER_INSERT(thread_id, SAMPLING_BUFFER(thread_id), evt);   \
	}                                                              \
}

#define TRACE_MISCEVENT(evttime,evttype,evtvalue,evtparam)        \
{                                                                 \
	event_t evt;                                                    \
	int thread_id = THREADID;                                       \
	if (tracejant && TracingBitmap[TASKID])                         \
	{                                                               \
		evt.time = evttime;                                           \
		evt.event = evttype;                                          \
		evt.value = evtvalue;                                         \
		evt.param.misc_param.param = (unsigned long long) (evtparam); \
		HARDWARE_COUNTERS_READ (thread_id, evt, FALSE);               \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);     \
	}                                                               \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_MISCEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
{                                                                 \
	event_t evt;                                                    \
	int thread_id = THREADID;                                       \
	if (tracejant && TracingBitmap[TASKID])                         \
	{                                                               \
		evt.time = evttime;                                           \
		evt.event = evttype;                                          \
		evt.value = evtvalue;                                         \
		evt.param.misc_param.param = (unsigned long long) (evtparam); \
		HARDWARE_COUNTERS_READ (thread_id, evt, TRUE);                \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);     \
	}                                                               \
}
#else
#define TRACE_MISCEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) TRACE_MISCEVENT(evttime,evttype,evtvalue,evtparam)
#endif

# define TRACE_N_MISCEVENT(evttime,count,evttypes,evtvalues,evtparams)        \
{                                                                             \
	if (tracejant && TracingBitmap[TASKID])                                     \
	{                                                                           \
		unsigned _i, thread_id=THREADID;                                          \
		event_t evts[count];                                                      \
                                                                              \
		for (_i=0; _i<count; _i++)                                                \
		{                                                                         \
			evts[_i].time = evttime;                                                \
			evts[_i].event = evttypes[_i];                                          \
			evts[_i].value = evtvalues[_i];                                         \
			evts[_i].param.misc_param.param = (unsigned long long) (evtparams[_i]); \
			HARDWARE_COUNTERS_READ(thread_id, evts[_i], FALSE);                     \
		}                                                                         \
		BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), evts, count);       \
	}                                                                           \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_N_MISCEVENTANDCOUNTERS(evttime,count,evttypes,evtvalues,evtparams) \
{                                                                                \
	if (tracejant && TracingBitmap[TASKID] && count > 0)                           \
	{                                                                              \
		unsigned _i, thread_id=THREADID;                                             \
		event_t evts[count];                                                         \
                                                                                 \
		for (_i=0; _i<count; _i++)                                                   \
		{                                                                            \
			evts[_i].time = evttime;                                                   \
			evts[_i].event = evttypes[_i];                                             \
			evts[_i].value = evtvalues[_i];                                            \
			evts[_i].param.misc_param.param = (unsigned long long) (evtparams[_i]);    \
			HARDWARE_COUNTERS_READ(thread_id, evts[_i], _i==0);                        \
		}                                                                            \
		BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), evts, count);          \
	}                                                                              \
}
#else
#define TRACE_N_MISCEVENTANDCOUNTERS(evttime,count,evttypes,evtvalues,evtparams) \
	{ \
		TRACE_N_MISCEVENT(evttime,count,evttypes,evtvalues,evtparams) \
	}
#endif

#define TRACE_EVENT(evttime,evttype,evtvalue)                 \
{                                                             \
	event_t evt;                                                \
	int thread_id = THREADID;                                   \
	if (tracejant && TracingBitmap[TASKID] )                    \
	{                                                           \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		evt.value = evtvalue;                                     \
		HARDWARE_COUNTERS_READ (thread_id, evt, FALSE);           \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_EVENTANDCOUNTERS(evttime,evttype,evtvalue,hwc_filter) \
{                                                                   \
	event_t evt;                                                      \
	int thread_id = THREADID;                                         \
	if (tracejant && TracingBitmap[TASKID] )                          \
	{                                                                 \
		evt.time = evttime;                                             \
		evt.event = evttype;                                            \
		evt.value = evtvalue;                                           \
		HARDWARE_COUNTERS_READ (thread_id, evt, hwc_filter);            \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);       \
	}                                                                 \
}

#define TRACE_EVENTAND_ACCUMULATEDCOUNTERS(evttime,evttype,evtvalue) \
{ \
	event_t evt;                                                      \
	int thread_id = THREADID;                                         \
	if (tracejant && TracingBitmap[TASKID] )                          \
	{                                                                 \
		COPY_ACCUMULATED_COUNTERS_HERE(thread_id, evt);                 \
		evt.time = evttime;                                             \
		evt.event = evttype;                                            \
		evt.value = evtvalue;                                           \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);       \
	} \
}
#else
#define TRACE_EVENTANDCOUNTERS(evttime,evttype,evtvalue,hwc_filter) \
	{ \
		TRACE_EVENT(evttime,evttype,evtvalue); \
	}

#define TRACE_EVENTAND_ACCUMULATEDCOUNTERS(evttime,evttype,evtvalue) \
	{ \
		TRACE_EVENT(evttime,evttype,evtvalue); \
	}
#endif

#define THREADS_TRACE_EVENT(from,to,evttime,evttype,evtvalue)   \
{                                                               \
	int thread_id;                                                \
	event_t evt;                                                  \
	if (tracejant && TracingBitmap[TASKID])                       \
	{                                                             \
		for (thread_id=from; thread_id<to; thread_id++)             \
		{                                                           \
			evt.time = evttime;                                       \
			evt.event = evttype;                                      \
			evt.value = evtvalue;                                     \
			HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);            \
			BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
		}                                                           \
	}                                                             \
}

#define THREAD_TRACE_MISCEVENT(thread,evttime,evttype,evtvalue,evtparam) \
{                                                               \
	event_t evt;                                                  \
	if (tracejant && TracingBitmap[TASKID])                       \
	{                                                             \
		evt.time = evttime;                                   \
		evt.event = evttype;                                  \
		evt.value = evtvalue;                                 \
		evt.param.misc_param.param = (unsigned long long) (evtparam); \
		HARDWARE_COUNTERS_READ(thread, evt, FALSE);           \
		BUFFER_INSERT(thread, TRACING_BUFFER(thread), evt);\
	}                                                             \
}

#define TRACE_EVENT_AND_GIVEN_COUNTERS(evttime, evttype, evtvalue, nc, counters)\
{                                                             \
	event_t evt;                                                \
	int _i, thread_id = THREADID;                               \
	if (tracejant && TracingBitmap[TASKID])                     \
	{                                                           \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		evt.value = evtvalue;                                     \
		for (_i=0; _i<nc; _i++)                                   \
			evt.HWCValues[_i] = (counters[_i]==NO_COUNTER)?NO_COUNTER:((INT64)counters[_i])&0xFFFFFFFF; \
		MARK_SET_READ(thread_id, evt, FALSE);                     \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
}

#define TRACE_N_EVENTS(evttime, count, evttypes, evtvalues)             \
{                                                                       \
  if (tracejant && TracingBitmap[TASKID] )                              \
  {                                                                     \
    unsigned _i, thread_id = THREADID;                                  \
    event_t evts[count];                                                \
                                                                        \
    for (_i=0; _i<count; _i++)                                          \
    {                                                                   \
      evts[_i].time = evttime;                                          \
      evts[_i].event = evttypes[_i];                                    \
      evts[_i].value = evtvalues[_i];                                   \
      HARDWARE_COUNTERS_READ (thread_id, evts[_i], FALSE);              \
    }                                                                   \
    BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), evts, count); \
  }                                                                     \
}

#define THREAD_TRACE_N_EVENTS(thread_id, evttime, count, evttypes, evtvalues) \
{                                                                             \
  if (tracejant && TracingBitmap[TASKID] )                                    \
  {                                                                           \
    int _i;                                                                   \
    event_t evts[count];                                                      \
                                                                              \
    for (_i=0; _i<count; _i++)                                                \
    {                                                                         \
      evts[_i].time = evttime;                                                \
      evts[_i].event = evttypes[_i];                                          \
      evts[_i].value = evtvalues[_i];                                         \
      HARDWARE_COUNTERS_READ (thread_id, evts[_i], FALSE);                    \
    }                                                                         \
    BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), evts, count);       \
  }                                                                           \
}

#define TRACE_USER_COMMUNICATION_EVENT(evttime,evttype,evtpartner,evtsize,evttag,id) \
{ \
	event_t evt; \
	int thread_id = THREADID; \
	if (tracejant) \
	{ \
		evt.time = (evttime); \
		evt.event = (evttype); \
		evt.value = 0; \
		evt.param.mpi_param.target = (long) (evtpartner); \
		evt.param.mpi_param.size = (evtsize); \
		evt.param.mpi_param.tag = (evttag); \
		evt.param.mpi_param.aux = (id); \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);  \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	} \
}

#define THREAD_TRACE_SAME_N_USER_COMMUNICATION_EVENT(thread,evttime,count,evttype,evtpartner,evtsize,evttag,id) \
{ \
	event_t evts[count]; \
	unsigned _i; \
	if (tracejant) \
	{ \
		for (_i=0; _i<count; _i++) \
		{ \
			evts[_i].time = (evttime); \
			evts[_i].event = (evttype); \
			evts[_i].value = 0; \
			evts[_i].param.mpi_param.target = (long) (evtpartner); \
			evts[_i].param.mpi_param.size = (evtsize); \
			evts[_i].param.mpi_param.tag = (evttag); \
			evts[_i].param.mpi_param.aux = (id); \
			HARDWARE_COUNTERS_READ(thread, evts[_i], FALSE);  \
		} \
		BUFFER_INSERT_N(thread, TRACING_BUFFER(thread), evts, count); \
	} \
}

#define THREAD_TRACE_USER_COMMUNICATION_EVENT(thread,evttime,evttype,evtpartner,evtsize,evttag,id) \
{ \
	event_t evt; \
	if (tracejant) \
	{ \
		evt.time = (evttime); \
		evt.event = (evttype); \
		evt.value = 0; \
		evt.param.mpi_param.target = (long) (evtpartner); \
		evt.param.mpi_param.size = (evtsize); \
		evt.param.mpi_param.tag = (evttag); \
		evt.param.mpi_param.aux = (id); \
		HARDWARE_COUNTERS_READ(thread, evt, FALSE);  \
		BUFFER_INSERT(thread, TRACING_BUFFER(thread), evt); \
	} \
}

#define TRACE_COUNTER(evttime,evttype)                        \
{                                                             \
	int thread_id = THREADID;                                   \
	event_t evt;                                                \
	if (tracejant && TracingBitmap[TASKID] )                    \
	{                                                           \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);            \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
}


#endif
