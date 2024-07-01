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

#ifndef TRACE_MACROS_MPI_H_INCLUDED
#define TRACE_MACROS_MPI_H_INCLUDED

#if defined(HAVE_BURST)
# include "burst_mode.h"
#endif


#define TRACE_MPI_CALLER_IS_ENABLED	(Trace_Caller_Enabled[CALLER_MPI])

#define TRACE_MPI_CALLER(evttime,evtvalue,offset)    \
{                                                    \
	if ( ( TRACE_MPI_CALLER_IS_ENABLED ) &&          \
	     ( Caller_Count[CALLER_MPI]>0 )  &&          \
	     ( evtvalue == EVT_BEGIN ) )                 \
	{                                                \
		Extrae_trace_callers (evttime, offset, CALLER_MPI); \
	}                                                \
}

#define TRACE_MPIINITEV(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                                           \
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
	BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
}

#define TRACE_MPIEVENT(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{ \
	if (tracejant) \
	{ \
		int thread_id = THREADID;                                                          \
		iotimer_t _tmp_current_time = evttime;                                             \
		                                                                                   \
		if (CURRENT_TRACE_MODE(thread_id) == TRACE_MODE_BURST)                            \
    {                                                                                  \
			BURSTS_MODE_TRACE_MPIEVENT( _tmp_current_time, evtvalue, FOUR_CALLS_AGO);        \
		}                                                                                  \
		else                                                                               \
		{                                                                                  \
			NORMAL_MODE_TRACE_MPIEVENT(thread_id,                                            \
			                           _tmp_current_time,                                    \
			                           evttype,                                              \
			                           evtvalue,                                             \
			                           evttarget,                                            \
			                           evtsize,                                              \
			                           evttag,                                               \
			                           evtcomm,                                              \
			                           evtaux,                                               \
			                           FOUR_CALLS_AGO);                                      \
		}                                                                                  \
		/* Check for pending changes */                                                    \
		if (evtvalue == EVT_BEGIN)                                                         \
		{                                                                                  \
			INCREASE_MPI_DEEPNESS(thread_id);                                                \
			last_mpi_begin_time = _tmp_current_time;                                         \
		}                                                                                  \
		else if (evtvalue == EVT_END)                                                      \
		{                                                                                  \
			DECREASE_MPI_DEEPNESS(thread_id);                                                \
		                                                                                   \
			/* Update last parallel region time */                                           \
			last_mpi_exit_time = _tmp_current_time;                                          \
		} \
	} \
}

#define NORMAL_MODE_TRACE_MPIEVENT(thread_id,evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux,offset) \
{                                                               \
	event_t evt;                                                  \
	int traceja_event = 0;                                        \
                                                                \
	if (tracejant_mpi)                                            \
	{                                                             \
		/* We don't want the compiler to reorganize ops */          \
		/* The "if" prevents that */                                \
		traceja_event = TracingBitmap[TASKID];                      \
		if ((TRACING_BITMAP_VALID_EVTYPE(evttype)) &&               \
		    (TRACING_BITMAP_VALID_EVTARGET(evttarget)))             \
		{                                                           \
			traceja_event |= TracingBitmap[((long)evttarget)];        \
		}                                                           \
		if (traceja_event)                                          \
		{                                                           \
			evt.time = (evttime);                                     \
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
				/* This happens once when the tracing mode changes */   \
				/* from CPU Bursts to Normal */                         \
				ADD_ACCUMULATED_COUNTERS_HERE(thread_id, evt);          \
				ACCUMULATED_COUNTERS_RESET(thread_id);                  \
			}                                                         \
			BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
			TRACE_MPI_CALLER (evt.time,evtvalue,offset)               \
		}                                                           \
	}                                                             \
}

#if defined(HAVE_BURST)
# define BURSTS_MODE_TRACE_MPIEVENT(evttime, evtvalue, offset)            \
{                                                                         \
  if (evtvalue == EVT_BEGIN)                                              \
  {                                                                       \
    if (xtr_burst_end()) TRACE_MPI_CALLER(evttime,evtvalue,offset);       \
  }                                                                       \
  else                                                                    \
  {                                                                       \
    xtr_burst_begin();                                                    \
  }                                                                       \
}
#else
# define BURSTS_MODE_TRACE_MPIEVENT(evttime, evtvalue, offset)
#endif

/***
Macro TRACE_MPIEVENT_NOHWC is used to trace: IRECVED_EV, PERSISTENT_REQ_EV.
Which are useless in the CPU Bursts tracing mode.
We conditionally check the tracing mode so as not to trace those while in CPU Bursts tracing mode.
***/
#define TRACE_MPIEVENT_NOHWC(evttime,evttype,evtvalue,evttarget,evtsize,evttag,evtcomm,evtaux) \
{                                                         \
   if (CURRENT_TRACE_MODE(THREADID) != TRACE_MODE_BURST) \
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
{                                                               \
	event_t evt;                                                  \
	int traceja_event = 0;                                        \
	int thread_id = THREADID;                                     \
                                                                \
	if (tracejant && tracejant_mpi)                               \
	{                                                             \
		/* We don't want the compiler to reorganize ops */          \
		/* The "if" prevents that */                                \
		traceja_event = TracingBitmap[TASKID];                      \
		if ((TRACING_BITMAP_VALID_EVTYPE(evttype)) &&               \
		    (TRACING_BITMAP_VALID_EVTARGET(evttarget)))             \
		{                                                           \
			traceja_event |= TracingBitmap[((int)evttarget)];         \
		}                                                           \
		if (traceja_event)                                          \
		{                                                           \
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
		}                                                           \
	}                                                             \
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
	HARDWARE_COUNTERS_READ (thread_id, evt, FALSE /* TRACING_HWC_MPI */); \
	BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
}

#endif /* TRACE_MACROS_MPI_H_INCLUDED */
