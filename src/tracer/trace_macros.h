/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

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
# define TRACING_BITMAP_VALID_EVTYPE(evttype)                                  \
   ((evttype) == BSEND_EV || (evttype) == SSEND_EV || (evttype) == RSEND_EV || \
    (evttype) == SEND_EV  || (evttype) == IRECVED_EV || (evttype) == RECV_EV)
# define TRACING_BITMAP_VALID_EVTARGET(evttarget)                              \
    (((long)evttarget) != MPI_ANY_SOURCE && ((long)evttarget) != MPI_PROC_NULL) 
#else
# define TRACING_BITMAP_VALID_EVTYPE(evttype)     (TRUE)
# define TRACING_BITMAP_VALID_EVTARGET(evttarget) (TRUE)
#endif /* MPI_SUPPORT */

#define TRACE_MPI_CALLER_IS_ENABLED	(Trace_Caller_Enabled[CALLER_MPI])

#define TRACE_MPI_CALLER(evttime,evtvalue,offset)    \
{                                                    \
	if ( ( TRACE_MPI_CALLER_IS_ENABLED ) &&          \
	     ( Caller_Count[CALLER_MPI]>0 )  &&          \
	     ( evtvalue == EVT_BEGIN ) )                 \
	{                                                \
		trace_callers (evttime, offset, CALLER_MPI); \
	}                                                \
}

#define SAMPLE_EVENT_HWC(evttime,evttype,evtvalue)                     \
{                                                                      \
	event_t evt;                                                       \
	int thread_id = THREADID;                                          \
	if (EnabledSampling)                                               \
	{                                                                  \
		if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)))               \
		{                                                              \
			evt.time = (evttime);                                      \
			evt.event = (evttype);                                     \
			evt.value = (evtvalue);                                    \
			/* We don't read counters right now */                     \
			HARDWARE_COUNTERS_READ(thread_id, evt, TRUE);              \
			BUFFER_INSERT(thread_id, SAMPLING_BUFFER(thread_id), evt); \
		}                                                              \
	}                                                                  \
}

#define SAMPLE_EVENT_NOHWC(evttime,evttype,evtvalue)                   \
{                                                                      \
	event_t evt;                                                       \
	int thread_id = THREADID;                                          \
	if (EnabledSampling)                                               \
	{                                                                  \
        if (!Buffer_IsFull (SAMPLING_BUFFER(thread_id)))               \
		{                                                              \
			evt.time = (evttime);                                      \
			evt.event = (evttype);                                     \
			evt.value = (evtvalue);                                    \
			HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);             \
			BUFFER_INSERT(thread_id, SAMPLING_BUFFER(thread_id), evt); \
		}                                                              \
	}                                                                  \
}

#define TRACE_MPIINITEV(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                                             \
	event_t evt;                                              \
	int thread_id = THREADID;                                 \
                                                              \
	evt.time = (evttime);                                     \
	evt.event = (evttype);                                    \
	evt.value = (evtvalue);                                   \
	evt.param.mpi_param.target = (evttarget);                 \
	evt.param.mpi_param.size = (evtsize);                     \
	evt.param.mpi_param.tag = (evttag);                       \
	evt.param.mpi_param.comm = (evtcomm);                     \
	evt.param.mpi_param.aux = (evtaux);                       \
	/* HWC 1st read */                                        \
	HARDWARE_COUNTERS_READ(thread_id, evt, TRUE);             \
	ACCUMULATED_COUNTERS_RESET(thread_id);                    \
	BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	if (evtvalue == EVT_END)                                  \
	{                                                         \
		last_mpi_exit_time = evt.time;                        \
	}                                                         \
}

#define TRACE_MPIEVENT(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                                                                        \
	int thread_id = THREADID;                                                            \
	unsigned long long current_time = evttime;                                           \
                                                                                         \
	if (CURRENT_TRACE_MODE(thread_id) == TRACE_MODE_BURSTS)                              \
	{                                                                                    \
		BURSTS_MODE_TRACE_MPIEVENT(thread_id, current_time, evtvalue);                   \
	}                                                                                    \
	else                                                                                 \
	{                                                                                    \
		NORMAL_MODE_TRACE_MPIEVENT(thread_id,                                            \
		                           current_time,                                         \
		                           evttype,                                              \
		                           evtvalue,                                             \
		                           evttarget,                                            \
		                           evtsize,                                              \
		                           evttag,                                               \
		                           evtcomm,                                              \
		                           evtaux,                                               \
		                           FOUR_CALLS_AGO);                                      \
	}                                                                                    \
	/* Check for pending changes */                                                      \
	if (evtvalue == EVT_BEGIN)                                                           \
	{                                                                                    \
		INCREASE_MPI_DEEPNESS(thread_id);                                                \
		last_mpi_begin_time = current_time;                                              \
                                                                                         \
        HARDWARE_COUNTERS_CHANGE(current_time, CHANGE_TIME, thread_id);                  \
                                                                                         \
	}                                                                                    \
	else if (evtvalue == EVT_END)                                                        \
	{                                                                                    \
		DECREASE_MPI_DEEPNESS(thread_id);                                                \
                                                                                         \
		/* Trace mode won't change if this MPI is stacked */                             \
		if (PENDING_TRACE_MODE_CHANGE(thread_id) && MPI_IS_NOT_STACKED(thread_id))       \
		{                                                                                \
			Trace_Mode_Change(thread_id, current_time);                                  \
		}                                                                                \
                                                                                         \
		/* Update last parallel region time */                                           \
		last_mpi_exit_time = current_time;                                               \
		Elapsed_Time_In_MPI += last_mpi_exit_time - last_mpi_begin_time;                 \
	}                                                                                    \
}

#define NORMAL_MODE_TRACE_MPIEVENT(thread_id,evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux,offset) \
{                                                                     \
	event_t evt;                                                      \
	int traceja_event = 0;                                            \
                                                                      \
	if (tracejant && tracejant_mpi)                                   \
	{                                                                 \
        /* We don't want the compiler to reorganize ops */            \
        /* The "if" prevents that */                                  \
		traceja_event = TracingBitmap[TASKID];                        \
		if ((TRACING_BITMAP_VALID_EVTYPE(evttype)) &&                 \
		    (TRACING_BITMAP_VALID_EVTARGET(evttarget)))               \
		{                                                             \
			traceja_event |= TracingBitmap[((long)evttarget)];        \
		}                                                             \
		if (traceja_event)                                            \
		{                                                             \
			evt.time = (evttime);                                     \
			TRACE_MPI_CALLER (evt.time,evtvalue,offset)                \
			evt.event = (evttype);                                    \
			evt.value = (evtvalue);                                   \
			evt.param.mpi_param.target = (long) (evttarget);          \
			evt.param.mpi_param.size = (evtsize);                     \
			evt.param.mpi_param.tag = (evttag);                       \
			evt.param.mpi_param.comm = (long) (evtcomm);              \
			evt.param.mpi_param.aux = (long) (evtaux);                \
            HARDWARE_COUNTERS_READ(thread_id, evt, TRACING_HWC_MPI);  \
			if (ACCUMULATED_COUNTERS_INITIALIZED(thread_id))          \
			{                                                         \
				/* This happens once when the tracing mode changes */ \
				/* from CPU Bursts to Normal */                       \
				ADD_ACCUMULATED_COUNTERS_HERE(thread_id, evt);        \
				ACCUMULATED_COUNTERS_RESET(thread_id);                \
			}                                                         \
			BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
		}                                                             \
	}                                                                 \
}

#define BURSTS_MODE_TRACE_MPIEVENT(thread_id, evttime, evtvalue)      \
{                                                                     \
	event_t evt_entry, evt_exit;                                        \
	evt_entry.time = last_mpi_exit_time;                                \
	evt_entry.event = CPU_BURST_EV;                                      \
	evt_entry.value = evtvalue;                                          \
	evt_exit.time = evttime;                                            \
	evt_exit.event = CPU_BURST_EV;                                      \
	evt_exit.value = 0;                                          \
	if (evtvalue == EVT_BEGIN)                                        \
	{                                                                 \
		if ((evt_exit.time - last_mpi_exit_time) > MINIMUM_BURST_DURATION)  \
		{                                                             \
			if (ACCUMULATED_COUNTERS_INITIALIZED(thread_id))          \
			{                                                         \
				COPY_ACCUMULATED_COUNTERS_HERE(thread_id, evt_entry);       \
				ACCUMULATED_COUNTERS_RESET(thread_id);                \
			}                                                         \
			else                                                      \
			{                                                         \
				/* This happens once when the tracing mode changes */ \
				/* from Normal to CPU Bursts, and after MPIINIT_EV */ \
				HARDWARE_COUNTERS_READ (thread_id, evt_entry, FALSE);       \
			}                                                         \
			BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt_entry); \
			OMPItrace_MPI_stats_Wrapper (last_mpi_exit_time);         \
			BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt_exit); \
		}                                                             \
		else                                                          \
		{                                                             \
			HARDWARE_COUNTERS_ACCUMULATE(thread_id, evt_exit, 1);               \
		}                                                             \
	}                                                                 \
	else                                                              \
	{                                                                 \
		HARDWARE_COUNTERS_ACCUMULATE(thread_id, evt_exit, 1);                   \
	}                                                                 \
}

/***
Macro TRACE_MPIEVENT_NOHWC is used to trace: IRECVED_EV, PERSISTENT_REQ_EV.
Which are useless in the CPU Bursts tracing mode.
We conditionally check the tracing mode so as not to trace those while in CPU Bursts tracing mode.
***/
#define TRACE_MPIEVENT_NOHWC(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                                         \
   if (CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURSTS) \
   {                                                      \
      REAL_TRACE_MPIEVENT_NOHWC(evttime,                  \
                                evttype,                  \
                                evtvalue,                 \
                                evttarget,                \
                                evtsize,                  \
                                evttag,                   \
                                evtcomm,                  \
                                evtaux);                  \
   }                                                      \
}

/***
Macro FORCE_TRACE_MPIEVENT is used to trace: Communicator definitions.
While in CPU Bursts tracing mode we still trace these events, since we may change later on to normal mode.
***/
#define FORCE_TRACE_MPIEVENT(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                       \
   REAL_FORCE_TRACE_MPIEVENT(evttime,   \
                             evttype,   \
                             evtvalue,  \
                             evttarget, \
                             evtsize,   \
                             evttag,    \
                             evtcomm,   \
                             evtaux);   \
}

#define REAL_TRACE_MPIEVENT_NOHWC(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                                                     \
	event_t evt;                                                      \
	int traceja_event = 0;                                            \
	int thread_id = THREADID;                                         \
                                                                      \
	if (tracejant && tracejant_mpi)                                   \
	{                                                                 \
		/* We don't want the compiler to reorganize ops */            \
		/* The "if" prevents that */                                  \
		traceja_event = TracingBitmap[TASKID];                        \
		if ((TRACING_BITMAP_VALID_EVTYPE(evttype)) &&                 \
		    (TRACING_BITMAP_VALID_EVTARGET(evttarget)))               \
		{                                                             \
			traceja_event |= TracingBitmap[((int)evttarget)];         \
		}                                                             \
		if (traceja_event)                                            \
		{                                                             \
			evt.time = (evttime);                                     \
			evt.event = (evttype);                                    \
			evt.value = (long) (evtvalue);                            \
			evt.param.mpi_param.target = (long) (evttarget);          \
			evt.param.mpi_param.size = (evtsize);                     \
			evt.param.mpi_param.tag = (evttag);                       \
			evt.param.mpi_param.comm = (long) (evtcomm);              \
			evt.param.mpi_param.aux = (long) (evtaux);                \
			HARDWARE_COUNTERS_READ (thread_id, evt, FALSE);           \
			BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
		}                                                             \
	}                                                                 \
}

#define REAL_FORCE_TRACE_MPIEVENT(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                                             \
	event_t evt;                                              \
	int thread_id = THREADID;                                 \
                                                              \
	evt.time = evttime;                                       \
	evt.event = evttype;                                      \
	evt.value = evtvalue;                                     \
	evt.param.mpi_param.target = (evttarget);                 \
	evt.param.mpi_param.size = (evtsize);                     \
	evt.param.mpi_param.tag = (evttag);                       \
	evt.param.mpi_param.comm = (intptr_t)(evtcomm);           \
	evt.param.mpi_param.aux = (evtaux);                       \
	HARDWARE_COUNTERS_READ (thread_id, evt, TRACING_HWC_MPI); \
	BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
}

#define TRACE_MISCEVENT(evttime,evttype,evtvalue,evtparam)            \
{                                                                     \
	event_t evt;                                                      \
	int thread_id = THREADID;                                         \
	if (tracejant && TracingBitmap[TASKID])                           \
	{                                                                 \
		evt.time = evttime;                                           \
		evt.event = evttype;                                          \
		evt.value = evtvalue;                                         \
		evt.param.misc_param.param = (unsigned long long) (evtparam); \
		HARDWARE_COUNTERS_READ (thread_id, evt, FALSE);               \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);     \
	}                                                                 \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_MISCEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
{                                                                     \
    event_t evt;                                                      \
    int thread_id = THREADID;                                         \
    if (tracejant && TracingBitmap[TASKID])                           \
    {                                                                 \
        evt.time = evttime;                                           \
        evt.event = evttype;                                          \
        evt.value = evtvalue;                                         \
        evt.param.misc_param.param = (unsigned long long) (evtparam); \
		HARDWARE_COUNTERS_READ (thread_id, evt, TRUE);                \
        BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);     \
	}                                                                 \
}
#else
#define TRACE_MISCEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) TRACE_MISCEVENT(evttime,evttype,evtvalue,evtparam)
#endif

#define TRACE_N_MISCEVENT(evttime,count,evttypes,evtvalues,evtparams)                \
{                                                                                    \
  if (tracejant && TracingBitmap[TASKID])                                            \
  {                                                                                  \
    int i, thread_id=THREADID;                                                       \
	event_t events_list[MAX_MULTIPLE_EVENTS];                                        \
                                                                                     \
	for (i=0; i<count; i++)                                                          \
	{                                                                                \
		events_list[i].time = evttime;                                               \
		events_list[i].event = evttypes[i];                                          \
		events_list[i].value = evtvalues[i];                                         \
		events_list[i].param.misc_param.param = (unsigned long long) (evtparams[i]); \
		HARDWARE_COUNTERS_READ(thread_id, events_list[i], FALSE);                    \
	}                                                                                \
	BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), events_list, count);       \
  }                                                                                  \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_N_MISCEVENTANDCOUNTERS(evttime,count,evttypes,evtvalues,evtparams)     \
{                                                                                    \
  if (tracejant && TracingBitmap[TASKID])                                            \
  {                                                                                  \
    int i, thread_id=THREADID;                                                       \
    event_t events_list[MAX_MULTIPLE_EVENTS];                                        \
                                                                                     \
    for (i=0; i<count; i++)                                                          \
    {                                                                                \
        events_list[i].time = evttime;                                               \
        events_list[i].event = evttypes[i];                                          \
        events_list[i].value = evtvalues[i];                                         \
        events_list[i].param.misc_param.param = (unsigned long long) (evtparams[i]); \
		if (i == 0)                                                                  \
		{                                                                            \
	        HARDWARE_COUNTERS_READ(thread_id, events_list[i], TRUE);                 \
		}                                                                            \
		else                                                                         \
		{                                                                            \
	        HARDWARE_COUNTERS_READ(thread_id, events_list[i], FALSE);                \
		}                                                                            \
    }                                                                                \
    BUFFER_INSERT_N(thread_id, TRACING_BUFFER(thread_id), events_list, count);       \
  }                                                                                  \
}
#else
#define TRACE_N_MISCEVENTANDCOUNTERS(evttime,count,evttypes,evtvalues,evtparams) TRACE_N_MISCEVENT(evttime,count,evttypes,evtvalues,evtparams)
#endif

#define TRACE_EVENT(evttime,evttype,evtvalue)                     \
{                                                                 \
	event_t evt;                                                  \
	int thread_id = THREADID;                                     \
	if (tracejant && TracingBitmap[TASKID] )                      \
	{                                                             \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		evt.value = evtvalue;                                     \
		HARDWARE_COUNTERS_READ (thread_id, evt, FALSE);           \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                             \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_EVENTANDCOUNTERS(evttime,evttype,evtvalue,hwc_filter) \
{                                                                   \
	event_t evt;                                                    \
	int thread_id = THREADID;                                       \
	if (tracejant && TracingBitmap[TASKID] )                        \
	{                                                               \
		evt.time = evttime;                                         \
		evt.event = evttype;                                        \
		evt.value = evtvalue;                                       \
		HARDWARE_COUNTERS_READ (thread_id, evt, hwc_filter);        \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);   \
	}                                                               \
}
#else
#define TRACE_EVENTANDCOUNTERS(evttime,evttype,evtvalue,hwc_filter) TRACE_EVENT(evttime,evttype,evtvalue)
#endif

#define TRACE_OMPEVENT(evttime,evttype,evtvalue,evtparam)         \
{                                                                 \
	int thread_id = THREADID;                                     \
	event_t evt;                                                  \
	if (tracejant && TracingBitmap[TASKID] && tracejant_omp)      \
	{                                                             \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		evt.value = evtvalue;                                     \
		evt.param.omp_param.param = evtparam;                     \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);            \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                             \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_OMPEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
{                                                                    \
	int thread_id = THREADID;                                        \
	event_t evt;                                                     \
	if (tracejant && TracingBitmap[TASKID] && tracejant_omp)         \
	{                                                                \
		evt.time = (evttime);                                        \
		evt.event = (evttype);                                       \
		evt.value = (evtvalue);                                      \
		evt.param.omp_param.param = (evtparam);                      \
		HARDWARE_COUNTERS_READ(thread_id, evt, TRACING_HWC_OMP);     \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);    \
	}                                                                \
}
#else
#define TRACE_OMPEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
  TRACE_OMPEVENT(evttime,evttype,evtvalue,evtparam)
#endif

#define TRACE_COUNTER(evttime,evttype)                            \
{                                                                 \
	int thread_id = THREADID;                                     \
	event_t evt;                                                  \
	if (tracejant && TracingBitmap[TASKID] )                      \
	{                                                             \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);            \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                             \
}

#define THREADS_TRACE_EVENT(from,to,evttime,evttype,evtvalue)      \
{                                                                  \
   int thread_id;                                                  \
   event_t evt;                                                    \
   if (tracejant && TracingBitmap[TASKID])                         \
   {                                                               \
      for (thread_id=from; thread_id<to; thread_id++)              \
      {                                                            \
         evt.time = evttime;                                       \
         evt.event = evttype;                                      \
         evt.value = evtvalue;                                     \
         HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);            \
         BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
      }                                                            \
   }                                                               \
}

#define TRACE_EVENT_AND_GIVEN_COUNTERS(evttime, evttype, evtvalue, nc, counters)\
{                                                                               \
    event_t evt;                                                                \
    int i, thread_id = THREADID;                                                \
    if (tracejant && TracingBitmap[TASKID])                                     \
    {                                                                           \
        evt.time = evttime;                                                     \
        evt.event = evttype;                                                    \
        evt.value = evtvalue;                                                   \
        for (i=0; i<nc; i++)                                                    \
            evt.HWCValues[i] = counters[i];                                     \
        MARK_SET_READ(evt, FALSE);                                              \
        BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt);               \
    }                                                                           \
}

#endif
