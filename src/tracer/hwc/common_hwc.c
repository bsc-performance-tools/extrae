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

#include "common.h"

#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include "utils.h"
#include "events.h"
#include "clock.h"
#include "threadid.h"
#include "record.h"
#include "trace_macros.h"
#include "wrapper.h"
#include "stdio.h"
#include "xml-parse.h"
#include "common_hwc.h"
#include "misc_wrapper.h"
#if defined(ENABLE_PEBS_SAMPLING)
# include "sampling-intel-pebs.h"
#endif

/*------------------------------------------------ Global Variables ---------*/
int HWCEnabled = FALSE;           /* Have the HWC been started? */

#if !defined(SAMPLING_SUPPORT)
int Reset_After_Read = TRUE;
#else
int Reset_After_Read = FALSE;
#endif

/* XXX: This variable should be defined at the Extrae core level */
int Trace_HWC_Enabled = TRUE;     /* Global variable that allows the gathering of HWC information */

int *HWC_Thread_Initialized;

/* XXX: These buffers should probably be external to this module, 
   and HWC_Accum should receive the buffer as an I/O parameter. */
long long **Accumulated_HWC;
int *Accumulated_HWC_Valid; /* Marks whether Accumulated_HWC has valid values */

HWC_Set_Count_t *CommonHWCs = NULL; /* To keep track in how many sets each counter appears */
int              AllHWCs    = 0;    /* Count of all the different counters from all sets   */

/*------------------------------------------------ Static Variables ---------*/

#if defined(PAPI_COUNTERS) && !defined(PAPIv3)
# error "-DNEW_HWC_SYSTEM requires PAPI v3 support"
#endif

struct HWC_Set_t *HWC_sets = NULL;
unsigned long long HWC_current_changeat = 0;
unsigned long long * HWC_current_timebegin;
unsigned long long * HWC_current_glopsbegin;
enum ChangeType_t HWC_current_changetype = CHANGE_NEVER;
enum ChangeTo_t HWC_current_changeto = CHANGE_SEQUENTIAL;
int HWC_num_sets = 0;
int * HWC_current_set;

/**
 * Checks whether the module has been started and the HWC are counting
 * \return 1 if HWC's are enabled, 0 otherwise.
 */
int HWC_IsEnabled()
{
	return HWCEnabled;
}

/**
 * Returns the active counters set (0 .. n-1). 
 * \return The active counters set (0 .. n-1).
 */
int HWC_Get_Current_Set (int threadid)
{
	return HWC_current_set[threadid];
}

/*
 * Returns the total number of sets.
 * \return The total number of sets. 
 */
int HWC_Get_Num_Sets ()
{
	return HWC_num_sets;
}

/**
 * Returns the Paraver IDs for the counters of the given set.
 * \param set_id The set identifier.
 * \param io_HWCParaverIds Buffer where the Paraver IDs will be stored. 
 * \return The number of counters in the given set. 
 */

int HWC_Get_Set_Counters_Ids (int set_id, int **io_HWCIds)
{
	int i=0, num_counters=0;
	int *HWCIds=NULL;

	num_counters = HWC_sets[set_id].num_counters;
    
	xmalloc(HWCIds, MAX_HWC * sizeof(int));

	for (i=0; i<num_counters; i++)
		HWCIds[i] = HWC_sets[set_id].counters[i];

	for (i=num_counters; i<MAX_HWC; i++)
		HWCIds[i] = NO_COUNTER;

	*io_HWCIds = HWCIds;
	return num_counters;
}

#include "../../merger/paraver/HardwareCounters.h" /* XXX: Include should be moved to common files */
int HWC_Get_Set_Counters_ParaverIds (int set_id, int **io_HWCParaverIds)
{
	int i=0, num_counters=0;
	int *HWCIds=NULL;

	num_counters = HWC_Get_Set_Counters_Ids (set_id, &HWCIds);
	
	/* Convert PAPI/PMAPI Ids to Paraver Ids */
	for (i=0; i<num_counters; i++)
	{
#if defined(PMAPI_COUNTERS)
		HWCIds[i] = HWC_COUNTER_TYPE(i, HWCIds[i]);
#else
		HWCIds[i] = HWC_COUNTER_TYPE(HWCIds[i]);
#endif
	}

    *io_HWCParaverIds = HWCIds;
    return num_counters;
}

/* Returns the index in which is stored the given counter in a set */
int HWC_Get_Position_In_Set (int set_id, int hwc_id)
{
	int i = 0, num_counters = 0;

    num_counters = HWC_sets[set_id].num_counters;

	for (i=0; i<num_counters; i++)
	{
		int cur_hwc_id;
#if defined(PMAPI_COUNTERS)
		cur_hwc_id = HWC_COUNTER_TYPE(i, HWC_sets[set_id].counters[i]);
#else
		cur_hwc_id = HWC_COUNTER_TYPE(HWC_sets[set_id].counters[i]);
#endif
		if (cur_hwc_id == hwc_id) return i;
	}
	return -1;
}

/**
 * Stops the current set
 * \param thread_id The thread that changes the set. 
 */
void HWC_Stop_Current_Set (UINT64 time, int thread_id)
{
	/* If there are less than 2 sets, don't do anything! */
	if (HWC_num_sets > 0)
	{
		/* make sure we don't loose the current counter values */
		Extrae_counters_at_Time_Wrapper(time);

		/* Actually stop the counters */
		HWCBE_STOP_SET (time, HWC_current_set[thread_id], thread_id);
	}
}

/**
 * Resumes the current set
 * \param thread_id The thread that changes the set. 
 */
void HWC_Start_Current_Set (UINT64 countglops, UINT64 time, int thread_id)
{
	/* If there are less than 2 sets, don't do anything! */
	if (HWC_num_sets > 0)
	{
		/* Actually start the counters */
		HWCBE_START_SET (countglops, time, HWC_current_set[thread_id], thread_id);
	}
}

/**
 * Stops the current set and starts reading the next one.
 * \param thread_id The thread that changes the set. 
 */
void HWC_Start_Next_Set (UINT64 countglops, UINT64 time, int thread_id)
{

	/* If there are less than 2 sets, don't do anything! */
	if (HWC_num_sets > 1)
	{
		HWC_Stop_Current_Set (time, thread_id);
		
		/* Move to the next set */
		if (HWC_current_changeto == CHANGE_SEQUENTIAL)
			HWC_current_set[thread_id] = (HWC_current_set[thread_id] + 1) % HWC_num_sets;
		else if (HWC_current_changeto == CHANGE_RANDOM)
			HWC_current_set[thread_id] = random()%HWC_num_sets;

		HWC_Start_Current_Set (countglops, time, thread_id);
	}

#if defined(ENABLE_PEBS_SAMPLING) 
	Extrae_IntelPEBS_nextSampling();
#endif
}

/** 
 * Stops the current set and starts reading the previous one.
 * \param thread_id The thread that changes the set.
 */
void HWC_Start_Previous_Set (UINT64 countglops, UINT64 time, int thread_id)
{
	/* If there are less than 2 sets, don't do anything! */
	if (HWC_num_sets > 1)
	{
		HWC_Stop_Current_Set (time, thread_id);

		/* Move to the previous set */
		if (HWC_current_changeto == CHANGE_SEQUENTIAL)
			HWC_current_set[thread_id] = ((HWC_current_set[thread_id] - 1) < 0) ? (HWC_num_sets - 1) : (HWC_current_set[thread_id] - 1) ;
		else if (HWC_current_changeto == CHANGE_RANDOM)
			HWC_current_set[thread_id] = random()%HWC_num_sets;

		HWC_Start_Current_Set (countglops, time, thread_id);
	}

#if defined(ENABLE_PEBS_SAMPLING) 
	Extrae_IntelPEBS_nextSampling();
#endif
}

/** 
 * Changes the current set for the given thread, depending on the number of global operations.
 * \param count_glops Counts how many global operations have been executed so far 
 * \param time Timestamp where the set change is checked
 * \param thread_id The thread identifier.
 * \return 1 if the set is changed, 0 otherwise.
 */

static inline int CheckForHWCSetChange_GLOPS (UINT64 countglops, UINT64 time, int threadid)
{
	int ret = 0;

	if (HWC_current_changeat != 0)
	{
		if (HWC_current_glopsbegin[threadid] + HWC_current_changeat <= countglops)
		{
			HWC_Start_Next_Set (countglops, time, threadid);
			ret = 1;
		}
	}
	return ret;
}

/** 
 * Changes the current set for the given thread, depending on the time that has passed. 
 * \param time Timestamp where the set change is checked
 * \param thread_id The thread identifier.
 * \return 1 if the set is changed, 0 otherwise.
 */
static inline int CheckForHWCSetChange_TIME (UINT64 countglops, UINT64 time, int threadid)
{
	int ret = 0;

//	fprintf (stderr, "HWC_current_timebegin[%d]=%llu HWC_current_changeat=%llu time = %llu\n", THREADID, HWC_current_timebegin[threadid], HWC_current_changeat, time);

	if (HWC_current_timebegin[threadid] + HWC_current_changeat < time)
	{
		HWC_Start_Next_Set (countglops, time, threadid);
		ret = 1;
	}
	return ret;
}

/**
 * Checks for pending set changes of the given thread.
 * \param count_glops Counts how many global operations have been executed so far 
 * \param time Timestamp where the set change is checked
 * \param thread_id The thread identifier.
 * \return 1 if the set is changed, 0 otherwise.
 */
int HWC_Check_Pending_Set_Change (UINT64 countglops, UINT64 time, int thread_id)
{
	if (HWC_current_changetype == CHANGE_GLOPS)
		return CheckForHWCSetChange_GLOPS(countglops, time, thread_id);
	else if (HWC_current_changetype == CHANGE_TIME)
		return CheckForHWCSetChange_TIME(countglops, time, thread_id);
	else
		return 0;
}

/** 
 * Initializes the hardware counters module.
 * \param options Configuration options.
 */
void HWC_Initialize (int options)
{
	int num_threads = Backend_getMaximumOfThreads();

	HWC_current_set = (int *)malloc(sizeof(int) * num_threads);
	ASSERT(HWC_current_set != NULL, "Cannot allocate memory for HWC_current_set");
	memset (HWC_current_set, 0, sizeof(int) * num_threads);

	HWC_current_timebegin = (unsigned long long *)malloc(sizeof(unsigned long long) * num_threads);
	ASSERT(HWC_current_timebegin != NULL, "Cannot allocate memory for HWC_current_timebegin");

	HWC_current_glopsbegin = (unsigned long long *)malloc(sizeof(unsigned long long) * num_threads);
	ASSERT(HWC_current_glopsbegin != NULL, "Cannot allocate memory for HWC_current_glopsbegin");

	HWCBE_INITIALIZE(options);
}

/**
 * Deallocates memory allocated by the routines in this module
 */
void HWC_CleanUp (unsigned nthreads)
{
	unsigned u;

	if (HWC_num_sets > 0)
	{
		HWCBE_CLEANUP_COUNTERS_THREAD(nthreads);

		xfree (HWC_current_set);
		xfree (HWC_current_timebegin);
		xfree (HWC_current_glopsbegin);
		xfree (HWC_Thread_Initialized);
		xfree (Accumulated_HWC_Valid);
		for (u = 0; u < nthreads; u++)
		{
			xfree (Accumulated_HWC[u]);
		}
		xfree (Accumulated_HWC);
	}
}

/**
 * Starts reading counters.
 * \param num_threads Total number of threads.
 */
void HWC_Start_Counters (int num_threads, UINT64 time, int forked)
{
	int i;

	/* Allocate memory if this process has not been forked */
	if (!forked)
	{
		HWC_Thread_Initialized = (int *) malloc (sizeof(int) * num_threads);
		ASSERT(HWC_Thread_Initialized!=NULL, "Cannot allocate memory for HWC_Thread_Initialized!");

		/* Mark all the threads as uninitialized */
		for (i = 0; i < num_threads; i++)
			HWC_Thread_Initialized[i] = FALSE;

		Accumulated_HWC_Valid = (int *)malloc(sizeof(int) * num_threads);
		ASSERT(Accumulated_HWC_Valid!=NULL, "Cannot allocate memory for Accumulated_HWC_Valid");

		Accumulated_HWC = (long long **)malloc(sizeof(long long *) * num_threads);
		ASSERT(Accumulated_HWC!=NULL, "Cannot allocate memory for Accumulated_HWC");

		for (i = 0; i < num_threads; i++)
		{
			Accumulated_HWC[i] = (long long *)malloc(sizeof(long long) * MAX_HWC);
			ASSERT(Accumulated_HWC[i]!=NULL, "Cannot allocate memory for Accumulated_HWC");
			HWC_Accum_Reset(i);
		}

		if (HWC_num_sets <= 0)
			return;

		HWCEnabled = TRUE;
	}

	/* Init counters for thread 0 */
	HWCEnabled = HWCBE_START_COUNTERS_THREAD (time, 0, forked);

	/* Inherit hwc set change values from thread 0 */
	for (i = 1; i < num_threads; i++)
	{
/*
 * XXX This used to be uncommented. Sets the same counter set to all the threads
 * of a task. Commented to allow every thread to have a different set.

		HWC_current_set[i] = HWC_current_set[0];
*/
		HWC_current_timebegin[i] = HWC_current_timebegin[0];
		HWC_current_glopsbegin[i] = HWC_current_glopsbegin[0];
	}
}

/** 
 * Starts reading counters for new threads. 
 * \param old_num_threads Previous number of threads.
 * \param new_num_threads New number of threads.
 */
void HWC_Restart_Counters (int old_num_threads, int new_num_threads)
{
	int i;

#if defined(PAPI_COUNTERS)
	for (i = 0; i < HWC_num_sets; i++)
		HWCBE_PAPI_Allocate_eventsets_per_thread (i, old_num_threads, new_num_threads);
#endif

	HWC_Thread_Initialized = (int *) realloc (HWC_Thread_Initialized, sizeof(int) * new_num_threads);
	ASSERT(HWC_Thread_Initialized!=NULL, "Cannot reallocate memory for HWC_Thread_Initialized!");

	/* Mark all the threads as uninitialized */
	for (i = old_num_threads; i < new_num_threads; i++)
		HWC_Thread_Initialized[i] = FALSE;

	Accumulated_HWC_Valid = (int *) realloc (Accumulated_HWC_Valid, sizeof(int) * new_num_threads);
	ASSERT(Accumulated_HWC_Valid!=NULL, "Cannot reallocate memory for Accumulated_HWC_Valid");

	Accumulated_HWC = (long long **) realloc (Accumulated_HWC, sizeof(long long *) * new_num_threads);
	ASSERT(Accumulated_HWC!=NULL, "Cannot reallocate memory for Accumulated_HWC");

	for (i = old_num_threads; i < new_num_threads; i++)
	{
		Accumulated_HWC[i] = (long long *)malloc(sizeof(long long) * MAX_HWC);
		ASSERT(Accumulated_HWC[i]!=NULL, "Cannot reallocate memory for Accumulated_HWC");
		HWC_Accum_Reset(i);
	}

	HWC_current_set = (int *) realloc (HWC_current_set, sizeof(int) * new_num_threads);
	ASSERT(HWC_current_set!=NULL, "Cannot reallocate memory for HWC_current_set");

	HWC_current_timebegin = (unsigned long long *) realloc (HWC_current_timebegin, sizeof(unsigned long long) * new_num_threads);
	ASSERT(HWC_current_timebegin!=NULL, "Cannot reallocate memory for HWC_current_timebegin");

	HWC_current_glopsbegin = (unsigned long long *) realloc (HWC_current_glopsbegin, sizeof(unsigned long long) * new_num_threads);
	ASSERT(HWC_current_glopsbegin!=NULL, "Cannot reallocate memory for HWC_current_glopsbegin");

	for (i = old_num_threads; i < new_num_threads; i++)
	{
		HWC_current_set[i] = 0;
		HWC_current_timebegin[i] = 0;
		HWC_current_glopsbegin[i] = 0;
	}
}

/**
 * Parses the XML configuration and setups the sets distribution.
 * \param task_id The task identifier.
 * \param num_tasks Total number of tasks.
 * \param distribution The user defined distribution scheme.
 */
void
HWC_Parse_XML_Config (int task_id, int num_tasks, char *distribution)
{
	unsigned threadid = 0;

	/* Do this if we have more than 1 counter set */
	if (HWC_num_sets > 1)
	{
		if (strncasecmp (distribution, "random", 6) == 0)
		{
			int i;
			unsigned long long rset;

			unsigned seed = ((unsigned) LAST_READ_TIME);
			for (i = 0; i < task_id; i++) /* Add some randomness here */
				seed = (seed >> 1) ^ ~(num_tasks | task_id);
			srandom (seed);

			rset = random()%HWC_num_sets;

			HWC_current_changeto = CHANGE_RANDOM;

			for(threadid=0; threadid<Backend_getMaximumOfThreads(); threadid++) 
				HWC_current_set[threadid] = rset;

			if (task_id == 0)
				fprintf (stdout, PACKAGE_NAME": Starting distribution hardware counters set is established to 'random'\n");
		}
		else if (strncasecmp (distribution, "cyclic", 6) == 0)
		{
			/* Sets are distributed among tasks like:
			0 1 2 3 .. n-1 0 1 2 3 .. n-1  0 1 2 3 ... */
			for(threadid=0; threadid<Backend_getMaximumOfThreads(); threadid++) 
				HWC_current_set[threadid] = task_id % HWC_num_sets;

			if (task_id == 0)
				fprintf (stdout, PACKAGE_NAME": Starting distribution hardware counters set is established to 'cyclic'\n");
		}
		else if (strncasecmp (distribution, "thread-cyclic", 13) == 0)
		{
			unsigned maxThreads;
			/* Sets are distributed among threads like:
			0 1 2 3 .. n-1 0 1 2 3 .. n-1  0 1 2 3 ... */
			maxThreads = Backend_getMaximumOfThreads();
			for(threadid=0; threadid<maxThreads; threadid++)
			{
				HWC_current_set[threadid] = (maxThreads * task_id + threadid) % HWC_num_sets;
			}

			if (task_id == 0)
				fprintf (stdout, PACKAGE_NAME": Starting distribution hardware counters set is established to 'thread_cyclic'\n");
		}
		else if (strncasecmp (distribution, "block", 5) == 0)
		{
			/* Sets are distributed among tasks in a 
			0 0 0 0 .. 1 1 1 1 .... n-1 n-1 n-1 n-1  
			fashion */

			/* a/b rounded to highest is (a+b-1)/b */
			int BlockDivisor = (num_tasks+HWC_num_sets-1) / HWC_num_sets;
			for(threadid=0; threadid<Backend_getMaximumOfThreads(); threadid++) 
			{
				if (BlockDivisor > 0)
					HWC_current_set[threadid] = task_id / BlockDivisor;
				else
					HWC_current_set[threadid] = 0;
			}

			if (task_id == 0)
				fprintf (stdout, PACKAGE_NAME": Starting distribution hardware counters set is established to 'block'\n");
		}
		else
		{
			/* Did the user placed a fixed value? */
			int value = atoi (distribution);
			if (value == 0)
			{
				if (task_id == 0)
					fprintf (stderr, PACKAGE_NAME": Warning! Cannot identify '%s' as a valid starting distribution set on the CPU counters. Setting to the first one.\n", distribution);
				for(threadid=0; threadid<Backend_getMaximumOfThreads(); threadid++)
					HWC_current_set[threadid] = 0;
			}
			else
				for(threadid=0; threadid<Backend_getMaximumOfThreads(); threadid++)
					HWC_current_set[threadid] = (HWC_num_sets<value-1)?HWC_num_sets:value-1;
		}
	}
}

/**
 * Parses the environment variables configuration (intended for executions without XML support).
 * \param task_id The task identifier.
 */
void HWC_Parse_Env_Config (int task_id)
{
    int numofcounters;
    char **setofcounters;

    numofcounters = explode (getenv("EXTRAE_COUNTERS"), ",", &setofcounters);
    HWC_Add_Set (1, task_id, numofcounters, setofcounters, getenv("EXTRAE_COUNTERS_DOMAIN"), 0, 0, 0, NULL, 0);
}

/** 
 * Reads counters for the given thread and stores the values in the given buffer. 
 * \param tid The thread identifier.
 * \param time When did the event occurred (if so)
 * \param store_buffer Buffer where the counters will be stored.
 * \return 1 if counters were read successfully, 0 otherwise.
 */
int HWC_Read (unsigned int tid, UINT64 time, long long *store_buffer)
{
	int read_ok = FALSE, reset_ok = FALSE; 

	if (HWCEnabled)
	{
		if (!HWC_Thread_Initialized[tid])
			HWCBE_START_COUNTERS_THREAD(time, tid, FALSE);
		TOUCH_LASTFIELD( store_buffer );

		read_ok = HWCBE_READ (tid, store_buffer);
		reset_ok = (Reset_After_Read ? HWCBE_RESET (tid) : TRUE);
	}
	return (HWCEnabled && read_ok && reset_ok);
}

/**
 * Resets the counters of the given thread.
 * \param tid The thread identifier.
 * \return 1 if success, 0 otherwise.
 */
int HWC_Reset (unsigned int tid)
{
	return ((HWCEnabled) ? HWCBE_RESET (tid) : 0);
}

/**
 * Returns whether counters are reset after reads 
 * \return 1 if counters are reset, 0 otherwise
 */
int HWC_Resetting ()
{
	return Reset_After_Read;
}

/**
 * Accumulates the counters of the given thread in a buffer.
 * \param tid The thread identifier.
 * \param time When did the event occurred (if so)
 * \return 1 if success, 0 otherwise.
 */
int HWC_Accum (unsigned int tid, UINT64 time)
{
	int accum_ok = FALSE; 

	if (HWCEnabled)
	{
		if (!HWC_Thread_Initialized[tid])
			HWCBE_START_COUNTERS_THREAD(time, tid, FALSE);
		TOUCH_LASTFIELD( Accumulated_HWC[tid] );

#if defined(SAMPLING_SUPPORT)
		/* If sampling is enabled, the counters are always in "accumulate" mode
		   because PAPI_reset is not called */
		accum_ok = HWCBE_READ (tid, Accumulated_HWC[tid]);
#else
		accum_ok = HWCBE_ACCUM (tid, Accumulated_HWC[tid]);
#endif

		Accumulated_HWC_Valid[tid] = TRUE;
	}
	return (HWCEnabled && accum_ok);
}

/**
 * Sets to zero the counters accumulated for the given thread.
 * \param tid The thread identifier.
 * \return 1 if success, 0 otherwise.
 */
int HWC_Accum_Reset (unsigned int tid)
{
	if (HWCEnabled)
	{
		Accumulated_HWC_Valid[tid] = FALSE;
		memset(Accumulated_HWC[tid], 0, MAX_HWC * sizeof(long long));
		return 1;
	}
	else return 0;
}

/** Returns whether Accumulated_HWC contains valid values or not */
int HWC_Accum_Valid_Values (unsigned int tid) 
{
	return ( HWCEnabled ? Accumulated_HWC_Valid[tid] : 0 );
}

/** 
 * Copy the counters accumulated for the given thread to the given buffer.
 * \param tid The thread identifier.
 * \param store_buffer Buffer where the accumulated counters will be copied. 
 */ 
int HWC_Accum_Copy_Here (unsigned int tid, long long *store_buffer)
{
	if (HWCEnabled)
	{
		memcpy(store_buffer, Accumulated_HWC[tid], MAX_HWC * sizeof(long long));
		return 1;
	}
	else return 0;
}

/**
 * Add the counters accumulated for the given thread to the given buffer.
 * \param tid The thread identifier.
 * \param store_buffer Buffer where the accumulated counters will be added. 
 */
int HWC_Accum_Add_Here (unsigned int tid, long long *store_buffer)
{
	int i;
	if (HWCEnabled)
	{
		for (i=0; i<MAX_HWC; i++)
		{
			store_buffer[i] += 	Accumulated_HWC[tid][i];
		}
		return 1;
	}
	else return 0;
}

/**
 * Configures a new set of counters.
 */
int HWC_Add_Set (int pretended_set, int rank, int ncounters, char **counters,
	char *domain, char *change_at_globalops, char *change_at_time,
	int num_overflows, char **overflow_counters, unsigned long long *overflow_values)
{
  int i                = 0;
  int counters_per_set = 0;
  int new_set          = 0;

  counters_per_set = HWCBE_ADD_SET (pretended_set, rank, ncounters, counters, domain, change_at_globalops, change_at_time, num_overflows, overflow_counters, overflow_values);
  new_set          = HWC_Get_Num_Sets() - 1;

  /* Count for each counter in how many sets appears */
  for (i=0; i<counters_per_set; i++)
  {
    int j      = 0;
    int found  = 0;
    int hwc_id = HWC_sets[new_set].counters[i];

    while ((!found) && (j < AllHWCs))
    {
      if (CommonHWCs[j].hwc_id == hwc_id)
      {
        CommonHWCs[j].sets_count ++;
        found = 1;
      }
      j ++;
    }
    if (!found)
    {
      CommonHWCs = (HWC_Set_Count_t *)realloc(CommonHWCs, (AllHWCs + 1) * sizeof(HWC_Set_Count_t));
      if (CommonHWCs == NULL)
      {
        fprintf (stderr, PACKAGE_NAME": Error! Unable to get memory for CommonHWCs");
        exit(-1);
      } 
      CommonHWCs[ AllHWCs ].hwc_id     = hwc_id;
      CommonHWCs[ AllHWCs ].sets_count = 1;

      AllHWCs ++;
    }
  } 
  return counters_per_set;
}

/** 
 * Configures the hwc change time frequency (forces the change_type to CHANGE_TIME!)
 * \param set The HWC set to configure.
 * \param ns The new frequency (in ns).
 */
void HWC_Set_ChangeAtTime_Frequency (int set, unsigned long long ns)
{
	if ((set >= 0) && (set < HWC_Get_Num_Sets()) && (ns > 0))
	{
		HWC_sets[set].change_type = CHANGE_TIME;
		HWC_sets[set].change_at = ns;
	}
	HWC_current_changetype = CHANGE_TIME;
}


/**
 * Returns 1 if the counter specificed by its set_id and index is common to all sets; 0 otherwise.
 */
int HWC_IsCommonToAllSets(int set_id, int hwc_index)
{
  int i;
  int hwc_id = HWC_sets[set_id].counters[hwc_index];

  for (i=0; i<AllHWCs; i++)
  {
    if (CommonHWCs[i].hwc_id == hwc_id)
    {
      if (CommonHWCs[i].sets_count == HWC_Get_Num_Sets())
      {
        return 1;
      }
    }
  }
  return 0;
}

/**
 * Returns how many counters are common to all sets 
 */
int HWC_GetNumberOfCommonCounters(void)
{
  int i           = 0;
  int common_hwcs = 0;

  for (i=0; i<AllHWCs; i++)
  {
    if (CommonHWCs[i].sets_count == HWC_Get_Num_Sets())
    {
      common_hwcs ++;
    }
  }
  return common_hwcs;
}

