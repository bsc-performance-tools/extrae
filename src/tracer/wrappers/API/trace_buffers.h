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
#include "signals.h"

/* Don't like these externs -> Declare fetch functions in wrapper.c and include prototypes in wrapper.h ? */
extern Buffer_t **TracingBuffer;
extern Buffer_t **SamplingBuffer;

#if defined(__cplusplus)
extern "C" {
#endif
int Extrae_Flush_Wrapper (Buffer_t *buffer);
#if defined(__cplusplus)
}
#endif

#define TRACING_BUFFER(tid) TracingBuffer[tid]
#define SAMPLING_BUFFER(tid) SamplingBuffer[tid]

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

