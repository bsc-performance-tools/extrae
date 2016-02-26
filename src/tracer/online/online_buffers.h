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

#ifndef __ONLINE_BUFFERS_H__
#define __ONLINE_BUFFERS_H__

#include "buffers.h"
#include "events.h"
#include "online_events.h"

extern Buffer_t *OnlineBuffer;

#define ONLINE_BUFFER OnlineBuffer

/* Online events are emitted as a MISC_EV */
#define TRACE_ONLINE_EVENT(evttime, evttype, evtvalue) \
{                                                      \
        event_t evt;                                   \
        evt.time  = evttime;                           \
        evt.event = ONLINE_EV;                         \
        evt.value = evttype;                           \
        evt.param.misc_param.param = evtvalue;         \
        evt.HWCReadSet = 0;                            \
        Buffer_InsertSingle(OnlineBuffer, &evt);       \
}

#define TRACE_ONLINE_COUNTERS(evttime, hwcset, hwcvalues) \
{                                                                  \
	int i=0;                                                   \
	event_t evt;                                               \
	evt.time = evttime;                                        \
	evt.event = ONLINE_EV;                                     \
	evt.value = HWC_EV;                                        \
	evt.HWCReadSet = hwcset + 1;                               \
	for (i=0; i<MAX_HWC; i++)                                  \
	{                                                          \
		evt.HWCValues[i] = hwcvalues[i];                   \
	}                                                          \
        Buffer_InsertSingle(OnlineBuffer, &evt);                   \
}

#endif /* __ONLINE_BUFFERS_H__ */
