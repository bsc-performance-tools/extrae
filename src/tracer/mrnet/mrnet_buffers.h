#ifndef __MRNET_BUFFERS_H__
#define __MRNET_BUFFERS_H__

#include "buffers.h"
#include "events.h"

extern Buffer_t *MRNetBuffer;

#define TRACE_MRN_EVENT(evttime, evttype, evtvalue) \
{                                                   \
	event_t evt;                                    \
	evt.time = evttime;                             \
	evt.event = evttype;                            \
	evt.value = evtvalue;                           \
	evt.HWCReadSet = 0;                             \
	Buffer_InsertSingle(MRNetBuffer, &evt);         \
}

#endif /* __MRNET_BUFFERS_H__ */

