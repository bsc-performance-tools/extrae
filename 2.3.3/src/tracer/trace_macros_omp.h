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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef TRACE_MACROS_OMP_H_INCLUDED
#define TRACE_MACROS_OMP_H_INCLUDED

#define TRACE_OMPEVENT(evttime,evttype,evtvalue,evtparam)     \
{                                                             \
	int thread_id = THREADID;                                   \
	event_t evt;                                                \
	if (tracejant && TracingBitmap[TASKID] && tracejant_omp)    \
	{                                                           \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		evt.value = evtvalue;                                     \
		evt.param.omp_param.param = evtparam;                     \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);            \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_OMPEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
{                                                             \
	int thread_id = THREADID;                                   \
	event_t evt;                                                \
	if (tracejant && TracingBitmap[TASKID] && tracejant_omp)    \
	{                                                           \
		evt.time = (evttime);                                     \
		evt.event = (evttype);                                    \
		evt.value = (evtvalue);                                   \
		evt.param.omp_param.param = (evtparam);                   \
		HARDWARE_COUNTERS_READ(thread_id, evt, TRACING_HWC_OMP);  \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
}
#else
#define TRACE_OMPEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
  TRACE_OMPEVENT(evttime,evttype,evtvalue,evtparam)
#endif

#define TRACE_PTHEVENT(evttime,evttype,evtvalue,evtparam)     \
{                                                             \
	int thread_id = THREADID;                                   \
	event_t evt;                                                \
	if (tracejant && TracingBitmap[TASKID] && tracejant_pthread)\
	{                                                           \
		evt.time = evttime;                                       \
		evt.event = evttype;                                      \
		evt.value = evtvalue;                                     \
		evt.param.omp_param.param = evtparam;                     \
		HARDWARE_COUNTERS_READ(thread_id, evt, FALSE);            \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
}

#if USE_HARDWARE_COUNTERS
#define TRACE_PTHEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
{                                                             \
	int thread_id = THREADID;                                   \
	event_t evt;                                                \
	if (tracejant && TracingBitmap[TASKID] && tracejant_pthread)\
	{                                                           \
		evt.time = (evttime);                                     \
		evt.event = (evttype);                                    \
		evt.value = (evtvalue);                                   \
		evt.param.omp_param.param = (evtparam);                   \
		HARDWARE_COUNTERS_READ(thread_id, evt, TRACING_HWC_PTHREAD);  \
		BUFFER_INSERT(thread_id, TRACING_BUFFER(thread_id), evt); \
	}                                                           \
}
#else
#define TRACE_PTHEVENTANDCOUNTERS(evttime,evttype,evtvalue,evtparam) \
  TRACE_PTHEVENT(evttime,evttype,evtvalue,evtparam)
#endif

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

#endif /* TRACE_MACROS_OMP_H_INCLUDED */
