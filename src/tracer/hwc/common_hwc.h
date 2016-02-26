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

#ifndef __COMMON_HWC_H__
#define __COMMON_HWC_H__

#include <config.h>
#include "sampling-common.h"
#include "sampling-timer.h"
#include "num_hwc.h"
#include "hwc_version.h"
#include "hwc.h"

/*------------------------------------------------ Structures ---------------*/

struct HWC_Set_t
{
#if defined(PAPI_COUNTERS)
    int domain;
    int *eventsets;
#elif defined(PMAPI_COUNTERS)
    pm_prog_t pmprog;
    int group;
#endif
    int counters[MAX_HWC];
    int num_counters;
    unsigned long long change_at;
    enum ChangeType_t change_type;
#if defined(PAPI_SAMPLING_SUPPORT)
    long long *OverflowValue;
    int *OverflowCounter;
    int NumOverflows;
#endif
};

#define MAX_HWC_DESCRIPTION_LENGTH  256

typedef struct HWC_Definition_st
{
	unsigned event_code;
	char description[MAX_HWC_DESCRIPTION_LENGTH];
} HWC_Definition_t;

/* A pair of counter ID and in how many sets appears */
typedef struct
{
  int hwc_id;
  int sets_count;
} HWC_Set_Count_t;

/*------------------------------------------------ Global Variables  --------*/

extern int *HWC_Thread_Initialized;
extern struct HWC_Set_t *HWC_sets;
extern int HWC_num_sets;
extern unsigned long long HWC_current_changeat;
extern unsigned long long * HWC_current_timebegin;
extern unsigned long long * HWC_current_glopsbegin;
extern enum ChangeType_t HWC_current_changetype;
extern int * HWC_current_set;

/*------------------------------------------------ Useful macros ------------*/

/** 
 * Touch the last position of the given buffer in order to produce the page fault before counters are read.
 */
#define TOUCH_LASTFIELD( data ) ( data[MAX_HWC-1] = data[MAX_HWC-1] )

/**
 * Stores the requested counters in the given buffer.
 */
#define HARDWARE_COUNTERS_REQUESTED( nc, counters, values_ptr ) \
{                                                               \
	int cc;                                                     \
	                                                            \
	for (cc = 0; cc < nc; cc++)                                 \
	{                                                           \
		if ( HWCEnabled )                                       \
			values_ptr[cc] = (long long) (counters[cc]);        \
		else                                                    \
			values_ptr[cc] = (long long) (NO_COUNTER);          \
	}                                                           \
	for (cc=nc; cc<MAX_HWC; cc++)                               \
		values_ptr[cc] = (long long) (NO_COUNTER);              \
}

/*------------------------------------------------ Backends access layer ----*/

#if defined(PAPI_COUNTERS) /* -------------------- PAPI Backend -------------*/

# include "papi_hwc.h"

# define HWCBE_INITIALIZE(options) \
    HWCBE_PAPI_Initialize (options)

# define HWCBE_START_COUNTERS_THREAD(time, tid, forked) \
    HWCBE_PAPI_Init_Thread(time, tid, forked)

# define HWCBE_START_SET(glops, time, current_set, thread_id) \
    HWCBE_PAPI_Start_Set(glops, time, current_set, thread_id)

# define HWCBE_STOP_SET(time, current_set, thread_id)  \
    HWCBE_PAPI_Stop_Set(time, current_set, thread_id)

# define HWCBE_ADD_SET(pretended_set, rank, ncounters, counters, domain,   \
                       change_at_globalops, change_at_time, num_overflows, \
                       overflow_counters, overflow_values)                 \
    HWCBE_PAPI_Add_Set(pretended_set, rank, ncounters, counters, domain,   \
	                   change_at_globalops, change_at_time, num_overflows,   \
	                   overflow_counters, overflow_values)

# define HWCBE_READ(thread_id, store_buffer)  \
    HWCBE_PAPI_Read(thread_id, store_buffer)

# define HWCBE_RESET(thread_id) \
    HWCBE_PAPI_Reset(thread_id)

# define HWCBE_ACCUM(thread_id, store_buffer) \
    HWCBE_PAPI_Accum(thread_id, store_buffer)

# define HWCBE_CLEANUP_COUNTERS_THREAD(nthreads) \
		HWCBE_PAPI_CleanUp(nthreads)

#define HWCBE_GET_COUNTER_DEFINITIONS(count) \
    HWCBE_PAPI_GetCounterDefinitions(count)

#elif defined(PMAPI_COUNTERS) /* -------------------- PMAPI Backend ---------*/

# include "pmapi_hwc.h" 

# define HWCBE_INITIALIZE(options) \
    HWCBE_PMAPI_Initialize (options)

# define HWCBE_START_COUNTERS_THREAD(time, tid, forked) \
    HWCBE_PMAPI_Init_Thread(time, tid, forked)

# define HWCBE_START_SET(glops, time, current_set, thread_id) \
    HWCBE_PMAPI_Start_Set(glops, time, current_set, thread_id);

# define HWCBE_STOP_SET(time, current_set, thread_id) \
    HWCBE_PMAPI_Stop_Set(time, current_set, thread_id);

# define HWCBE_ADD_SET(pretended_set, rank, ncounters, counters, domain,    \
                       change_at_globalops, change_at_time, num_overflows,  \
                       overflow_counters, overflow_values)                  \
	HWCBE_PMAPI_Add_Set(pretended_set, rank, ncounters, counters, domain,     \
	                    change_at_globalops, change_at_time, num_overflows,   \
	                    overflow_counters, overflow_values)

# define HWCBE_READ(thread_id, store_buffer)  \
    HWCBE_PMAPI_Read(thread_id, store_buffer)

# define HWCBE_RESET(thread_id) 1

# define HWCBE_ACCUM(thread_id, store_buffer) 1

# define HWCBE_CLEANUP_COUNTERS_THREAD(nthreads) \
		HWCBE_PMAPI_CleanUp(nthreads)

#define HWCBE_GET_COUNTER_DEFINITIONS(count) \
    HWCBE_PMAPI_GetCounterDefinitions(count)

#endif

#endif /* __COMMON_HWC_H__ */

