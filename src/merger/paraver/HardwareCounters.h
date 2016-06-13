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

#ifndef _HARDWARE_COUNTERS_H
#define _HARDWARE_COUNTERS_H

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)

#include "num_hwc.h"
#include "hwc_version.h"
#include "record.h"
#include "papiStdEventDefs.h"

#if !USE_HARDWARE_COUNTERS
  /* Little things to have configured if PAPI cannot be used */
# define PAPI_NATIVE_MASK 0x40000000 
#endif

/* Macro per calcular el tipus que cal assignar al comptador.
 * Els comptadors a linux son numeros aixi: 0x800000XX
 * Els comptadors a linux(natiu) son numeros aixi: 0x40000XXX
 * per tant, nomes cal tenir en compte el byte mes baix per
 * no tenir numeros inmensos. */

#if defined(PAPI_COUNTERS)
# if defined(PAPIv2)
#  define HWC_COUNTER_TYPE(x) (HWC_BASE + (x & 0x000000FF))
# elif defined(PAPIv3)
#  define HWC_COUNTER_TYPE(x) \
		(x&PAPI_NATIVE_MASK)?(HWC_BASE_NATIVE + (x & 0x0000FFFF)):(HWC_BASE + (x & 0x0000FFFF))
# endif
#elif defined(PMAPI_COUNTERS)
# define HWC_COUNTER_TYPE(cnt,x) (HWC_BASE + cnt*1000 + x)
#else
# define HWC_COUNTER_TYPE(x) x
#endif

/*
 * CntQueue (type): structure to store the counter that has been readed during
 *                  the application execution.
 * FreeListItems  : Free CntQueue queue items list.
 * CountersTraced : Queue of CntQueue strucutures. Will contain a list of all
 *                  the conunters that has been traced during application
 *                  execution.
 */
typedef struct _cQueue
{
  struct _cQueue *next, *prev;

  int Events[MAX_HWC];
  int Traced[MAX_HWC];          /*
                                 * * Boolean field for each counter that
                                 * *  indicates if counter is readed or not.
                                 * *
                                 */
} CntQueue;

extern CntQueue CountersTraced;

int HardwareCounters_Emit (int ptask, int task, int thread,
	unsigned long long time, event_t * Event,
	int *outtype, unsigned long long *outvalue,
	int absolute);
void HardwareCounters_Show (const event_t * Event, int ncounters);
void HardwareCounters_Get (const event_t *Event, unsigned long long *buffer);
void HardwareCounters_NewSetDefinition (int ptask, int task, int thread, int newSet, long long *HWCIds);
int * HardwareCounters_GetSetIds(int ptask, int task, int thread, int set_id);
int HardwareCounters_GetCurrentSet(int ptask, int task, int thread);
void HardwareCounters_Change (int ptask, int task, int thread, int newSet, int *outtypes, unsigned long long *outvalues);
void HardwareCounters_SetOverflow (int ptask, int task, int thread, event_t *Event);

#if defined(PARALLEL_MERGE)
void Share_Counters_Usage (int size, int rank);
#endif

#endif /* USE_HARDWARE_COUNTERS */

#endif /* _HARDWARE_COUNTERS_H */
