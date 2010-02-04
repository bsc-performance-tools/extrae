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

#ifndef __TRACE_HWC_H__
#define __TRACE_HWC_H__

/* Interface to communicate the MPItrace core with the HWC module */

#if USE_HARDWARE_COUNTERS

# include "hwc.h"

# define MARK_SET_READ(tid, evt, filter)                                                   \
{                                                                                     \
	evt.HWCReadSet = ((filter && HWC_IsEnabled()) ? (HWC_Get_Current_Set(tid) + 1) : 0); \
}

/* Store counters values in the event and mark them as read */
# define HARDWARE_COUNTERS_READ(tid, evt, filter)                      \
{                                                                      \
	int read_ok = FALSE;                                               \
	if (filter && HWC_IsEnabled())                                     \
	{                                                                  \
		read_ok = HWC_Read (tid, evt.time, evt.HWCValues);             \
	}                                                                  \
	/* We write the counters even if there are errors while reading */ \
	MARK_SET_READ(tid, evt, read_ok);                                       \
} 

# define HARDWARE_COUNTERS_ACCUMULATE(tid, evt, filter)                \
{                                                                      \
    if (filter && HWC_IsEnabled())                                     \
    {                                                                  \
        HWC_Accum (tid, evt.time);                                     \
		/* XXX: Reset ACCUMULATED counters here? Very likely!!! */     \
    }                                                                  \
	/* We write the counters even if there are errors while reading */ \
	MARK_SET_READ(tid, evt, filter);                                       \
}

# define ACCUMULATED_COUNTERS_RESET(tid) HWC_Accum_Reset(tid)

# define ACCUMULATED_COUNTERS_INITIALIZED(tid) HWC_Accum_Valid_Values(tid)

/* Copy accumulated counters to the event and mark them as read */
# define COPY_ACCUMULATED_COUNTERS_HERE(tid, evt) \
{                                                 \
	HWC_Accum_Copy_Here(tid, evt.HWCValues);      \
	MARK_SET_READ(tid, evt, TRUE);                     \
}

/* Add accumulated counters to the event, but DON'T mark them as read. If we're adding
   is because there has been a previous read so they're already marked as read. */
# define ADD_ACCUMULATED_COUNTERS_HERE(tid, evt) HWC_Accum_Add_Here(tid, evt.HWCValues)

# define HARDWARE_COUNTERS_CHANGE(time, type, tid) HWC_Check_Pending_Set_Change(time, type, tid);

#else /* ! USE_HARDWARE_COUNTERS */

# define HARDWARE_COUNTERS_READ(tid, evt, filter)
# define HARDWARE_COUNTERS_ACCUMULATE(tid, evt, filter)
# define ACCUMULATED_COUNTERS_RESET(tid)
# define ACCUMULATED_COUNTERS_INITIALIZED(tid) 0
# define COPY_ACCUMULATED_COUNTERS_HERE(tid, evt)
# define ADD_ACCUMULATED_COUNTERS_HERE(tid, evt)
# define HARDWARE_COUNTERS_CHANGE(time, type, tid)

#endif

#endif /* __TRACE_HWC_H__ */
