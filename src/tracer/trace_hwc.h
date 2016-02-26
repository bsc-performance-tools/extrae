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

#ifndef __TRACE_HWC_H__
#define __TRACE_HWC_H__

/* Interface to communicate the Extrae core with the HWC module */

#if USE_HARDWARE_COUNTERS

# include "hwc.h"
#if defined(MPI_SUPPORT)
# include "mpi_interface.h"
#endif

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
	MARK_SET_READ(tid, evt, TRUE);                \
}

/* Add accumulated counters to the event, but DON'T mark them as read. If we're adding
   is because there has been a previous read so they're already marked as read. */
# define ADD_ACCUMULATED_COUNTERS_HERE(tid, evt) HWC_Accum_Add_Here(tid, evt.HWCValues)


#if defined(MPI_SUPPORT)
# define COUNT_GLOBAL_OPS Extrae_MPI_getNumOpsGlobals()
#else
# define COUNT_GLOBAL_OPS 0
#endif

# define HARDWARE_COUNTERS_CHANGE(time, tid) HWC_Check_Pending_Set_Change(COUNT_GLOBAL_OPS, time, tid);

#else /* ! USE_HARDWARE_COUNTERS */

# define HARDWARE_COUNTERS_READ(tid, evt, filter)
# define HARDWARE_COUNTERS_ACCUMULATE(tid, evt, filter)
# define ACCUMULATED_COUNTERS_RESET(tid)
# define ACCUMULATED_COUNTERS_INITIALIZED(tid) 0
# define COPY_ACCUMULATED_COUNTERS_HERE(tid, evt)
# define ADD_ACCUMULATED_COUNTERS_HERE(tid, evt)
# define HARDWARE_COUNTERS_CHANGE(time, tid) {}
# define MARK_SET_READ(tid, evt, filter)

#endif

#endif /* __TRACE_HWC_H__ */
