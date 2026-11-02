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

#ifndef __TRACE_BUFFERS_H__
#define __TRACE_BUFFERS_H__

#include "buffers.h"
#include "clock.h"
#include "signals.h"

/* Keep all buffer-related values for one tracing thread ID together. */
typedef struct
{
	Buffer_t *tracing_buffer;
	iotimer_t last_cpu_event_time;
	int last_cpu_id;
#if defined(SAMPLING_SUPPORT)
	Buffer_t *sampling_buffer;
#endif
} Extrae_thread_trace_data_t;

#if defined(__cplusplus)
extern "C" {
#endif
/* Return the tracing record for this ID, or NULL if it is unavailable. */
Extrae_thread_trace_data_t *Extrae_get_thread_trace_data (unsigned thread_id);
int Extrae_Flush_Wrapper (Buffer_t *buffer);
#if defined(__cplusplus)
}
#endif

#define TRACING_BUFFER(tid) (Extrae_get_thread_trace_data(tid)->tracing_buffer)
#define SAMPLING_BUFFER(tid) (Extrae_get_thread_trace_data(tid)->sampling_buffer)

#define BUFFER_INSERT(tid, buffer, event)                   \
{                                                           \
	Signals_Inhibit();                                      \
	Buffer_InsertSingle (buffer, &event);                   \
	Signals_Desinhibit();                                   \
	Signals_ExecuteDeferred();                              \
}
	
#define BUFFER_INSERT_N(tid, buffer, events_list, num_events)            \
{                                                                        \
	if (num_events > 0)                                                  \
	{                                                                    \
		Signals_Inhibit();                                               \
		Buffer_InsertMultiple(buffer, events_list, num_events);          \
		Signals_Desinhibit();                                            \
		Signals_ExecuteDeferred();                                       \
	}                                                                    \
}

#endif /* __TRACE_BUFFERS_H__ */

