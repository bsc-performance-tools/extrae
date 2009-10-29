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
int MPItrace_Flush_Wrapper(Buffer_t *buffer);
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

