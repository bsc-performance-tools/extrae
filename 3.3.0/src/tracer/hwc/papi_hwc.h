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

#ifndef __PAPI_HWC_H__
#define __PAPI_HWC_H__

#include "num_hwc.h"

/*------------------------------------------------ Prototypes ---------------*/

void HWCBE_PAPI_Initialize (int TRCOptions);
int HWCBE_PAPI_Init_Thread (UINT64 time, int threadid, int forked);
int HWCBE_PAPI_Allocate_eventsets_per_thread (int num_set, int old_thread_num, int new_thread_num);

int HWCBE_PAPI_Start_Set (UINT64 countglops, UINT64 time, int numset, int threadid);
int HWCBE_PAPI_Stop_Set (UINT64 time, int numset, int threadid);
int HWCBE_PAPI_Add_Set (int pretended_set, int rank, int ncounters, char **counters, char *domain, 
                      char *change_at_globalops, char *change_at_time, int num_overflows, 
                      char **overflow_counters, unsigned long long *overflow_values);

int HWCBE_PAPI_Read (unsigned int tid, long long *store_buffer);
int HWCBE_PAPI_Reset (unsigned int tid);
int HWCBE_PAPI_Accum (unsigned int tid, long long *store_buffer);

void HWCBE_PAPI_CleanUp (unsigned nthreads);

HWC_Definition_t *HWCBE_PAPI_GetCounterDefinitions(unsigned *count);

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
#define HWCEVTSET(tid) (HWC_sets[HWC_Get_Current_Set(tid)].eventsets[tid])

#endif /* __PAPI_HWC_H__ */

