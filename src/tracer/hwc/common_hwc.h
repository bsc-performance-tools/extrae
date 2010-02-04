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

#ifndef __COMMON_HWC_H__
#define __COMMON_HWC_H__

#include <config.h>
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

#if defined(SAMPLING_SUPPORT)
    long long *OverflowValue;
    int *OverflowCounter;
    int NumOverflows;
#endif
};

/*------------------------------------------------ Global Variables  --------*/

extern int *HWC_Thread_Initialized;
extern struct HWC_Set_t *HWC_sets;
extern int HWC_num_sets;
extern unsigned long long HWC_current_changeat;
extern unsigned long long * HWC_current_timebegin;
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

# define HWCBE_START_COUNTERS_THREAD(time, tid)    \
    HWCBE_PAPI_Init_Thread(time, tid)

# define HWCBE_START_SET(time, current_set, thread_id) \
    HWCBE_PAPI_Start_Set(time, current_set, thread_id)

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

#elif defined(PMAPI_COUNTERS) /* -------------------- PMAPI Backend ---------*/

# include "pmapi_hwc.h" 

# define HWCBE_INITIALIZE(options) \
    HWCBE_PMAPI_Initialize (options)

# define HWCBE_START_COUNTERS_THREAD(time, tid) \
    HWCBE_PMAPI_Init_Thread(time, tid)

# define HWCBE_START_SET(time, current_set, thread_id) \
    HWCBE_PMAPI_Start_Set(time, current_set, thread_id);

# define HWCBE_STOP_SET(time, current_set, thread_id) \
    HWCBE_PMAPI_Stop_Set(time, current_set, thread_id)

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

#endif

#endif /* __COMMON_HWC_H__ */

