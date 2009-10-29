/****************************************************************************\
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

#ifndef __PAPI_HWC_H__
#define __PAPI_HWC_H__

#include "num_hwc.h"

/*------------------------------------------------ Prototypes ---------------*/

void HWCBE_PAPI_Initialize (int TRCOptions);
int HWCBE_PAPI_Init_Thread (UINT64 time, int threadid);
int HWCBE_PAPI_Allocate_eventsets_per_thread (int num_set, int old_thread_num, int new_thread_num);

int HWCBE_PAPI_Start_Set (UINT64 time, int numset, int threadid);
int HWCBE_PAPI_Stop_Set (UINT64 time, int numset, int threadid);
int HWCBE_PAPI_Add_Set (int pretended_set, int rank, int ncounters, char **counters, char *domain, 
                      char *change_at_globalops, char *change_at_time, int num_overflows, 
                      char **overflow_counters, unsigned long long *overflow_values);

int HWCBE_PAPI_Read (unsigned int tid, long long *store_buffer);
int HWCBE_PAPI_Reset (unsigned int tid);
int HWCBE_PAPI_Accum (unsigned int tid, long long *store_buffer);

/*------------------------------------------------ Useful Macros ------------*/

/**
 * Stores which counters did overflow in the given buffer (?).
 */
#define HARDWARE_COUNTERS_OVERFLOW( nc, counters, no, counters_ovf, values_ptr ) \
{                                                                                \
	int found, cc, co;                                                           \
	                                                                             \
	for (cc = 0; cc < nc; cc++)                                                  \
	{                                                                            \
		for (co = 0, found = 0; co < no; co++)                                   \
			found |= counters[cc] == counters_ovf[co];                           \
		if (found)                                                               \
			values_ptr[cc] = (long long) (SAMPLE_COUNTER);                       \
		else                                                                     \
			values_ptr[cc] = (long long) (NO_COUNTER);                           \
	}                                                                            \
	for (cc = nc; cc < MAX_HWC; cc++)                                            \
		values_ptr[cc] = (long long) (NO_COUNTER);                               \
}

/**
 * Returns the EventSet of the given thread for the current set.
 */
#define HWCEVTSET(tid) (HWC_sets[HWC_Get_Current_Set()].eventsets[tid])

#endif /* __PAPI_HWC_H__ */

