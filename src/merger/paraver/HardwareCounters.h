/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/paraver/HardwareCounters.h,v $
 | 
 | @last_commit: $Date: 2008/12/01 10:39:14 $
 | @version:     $Revision: 1.7 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef _HARDWARE_COUNTERS_H
#define _HARDWARE_COUNTERS_H

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)

#include "num_hwc.h"
#include "hwc_version.h"
#include "record.h"

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
		(x&PAPI_NATIVE_MASK)?(HWC_BASE_NATIVE + (x & 0x000003FF)):(HWC_BASE + (x & 0x000003FF))
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

  long long Events[MAX_HWC];
  int Traced[MAX_HWC];          /*
                                 * * Boolean field for each counter that
                                 * *  indicates if counter is readed or not.
                                 * *
                                 */
}
CntQueue;

extern CntQueue CountersTraced;

void HardwareCounters_Emit (int cpu, int ptask, int task, int thread,
  long long time, event_t * Event, unsigned int *outtype,
  unsigned long long *outvalue);
void HardwareCounters_Show (event_t * Event);
void HardwareCounters_Get (event_t *Event, unsigned long long *buffer);
void HardwareCounters_Change (int cpu, int ptask, int task, int thread,
	event_t *current, unsigned long long time, unsigned int *outtypes,
	unsigned long long *outvalues);
void HardwareCounters_SetOverflow (int ptask, int task, int thread, event_t *Event);

#if defined(PARALLEL_MERGE)
void Share_Counters_Usage (int size, int rank);
#endif

#endif /* USE_HARDWARE_COUNTERS */

#endif /* _HARDWARE_COUNTERS_H */
