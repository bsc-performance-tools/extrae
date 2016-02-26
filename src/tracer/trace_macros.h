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
# define COMM_STATS_WRAPPER(x) \
    Extrae_MPI_stats_Wrapper (x);
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
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)))               \
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
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)))               \
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
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)))               \
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
	if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)))               \
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

#if defined(DCARRERA_HADOOP)
# define TRACE_N_MISCEVENT(evttime,count,evttypes,evtvalues,evtparams) \
{ \
	if (tracejant && TracingBitmap[TASKID]) \
	{ \
		unsigned i, thread_id=THREADID; \
		event_t evts[count]; \
\
		for (i=0; i<count; i++) \
		{ \
			evts[i].time = evttime; \
			evts[i].event = evttypes[i]; \
			evts[i].value = evtvalues[i]; \
			evts[i].param.misc_param.param = (unsigned long long) (evtparams[i]); \
			HARDWARE_COUNTERS_READ(thread_id, evts[i], FALSE); \
		} \
		BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), evts, count); \
	} \
}
#else
# define TRACE_N_MISCEVENT(evttime,count,evttypes,evtvalues,evtparams)              \
{                                                                                  \
	if (tracejant && TracingBitmap[TASKID])                                          \
	{                                                                                \
		unsigned i, thread_id=THREADID;                                                \
		event_t evts[count];                                                    \
                                                                                   \
		for (i=0; i<count; i++)                                                        \
		{                                                                              \
			evts[i].time = evttime;                                               \
			evts[i].event = evttypes[i];                                          \
			evts[i].value = evtvalues[i];                                         \
			evts[i].param.misc_param.param = (unsigned long long) (evtparams[i]); \
			HARDWARE_COUNTERS_READ(thread_id, evts[i], FALSE);                    \
		}                                                                              \
		BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), evts, count);     \
	}                                                                                \
}
#endif

#if USE_HARDWARE_COUNTERS
#define TRACE_N_MISCEVENTANDCOUNTERS(evttime,count,evttypes,evtvalues,evtparams)   \
{                                                                                  \
	if (tracejant && TracingBitmap[TASKID] && count > 0)                             \
	{                                                                                \
		unsigned i, thread_id=THREADID;                                                \
		event_t evts[count];                                                    \
                                                                                   \
		for (i=0; i<count; i++)                                                        \
		{                                                                              \
			evts[i].time = evttime;                                               \
			evts[i].event = evttypes[i];                                          \
			evts[i].value = evtvalues[i];                                         \
			evts[i].param.misc_param.param = (unsigned long long) (evtparams[i]); \
			HARDWARE_COUNTERS_READ(thread_id, evts[i], i==0);                     \
		}                                                                              \
		BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), evts, count);     \
	}                                                                                \
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
#else
#define TRACE_EVENTANDCOUNTERS(evttime,evttype,evtvalue,hwc_filter) \
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
	int i, thread_id = THREADID;                                \
	if (tracejant && TracingBitmap[TASKID])                     \
	{                                                           \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		evt.value = evtvalue;                                     \
		for (i=0; i<nc; i++)                                      \
			evt.HWCValues[i] = (counters[i]==NO_COUNTER)?NO_COUNTER:((INT64)counters[i])&0xFFFFFFFF; \
		MARK_SET_READ(thread_id, evt, FALSE);                     \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
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
