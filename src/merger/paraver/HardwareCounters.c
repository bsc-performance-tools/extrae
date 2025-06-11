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

#if USE_HARDWARE_COUNTERS || defined(HETEROGENEOUS_SUPPORT)

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
# include <sys/stat.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif
#if defined(PARALLEL_MERGE)
# include <mpi.h>
#endif
#include "queue.h"
#include "events.h"
#include "object_tree.h"
#include "HardwareCounters.h"
#include "trace_to_prv.h"
#include "paraver_generator.h"
#include "dimemas_generator.h"
#include "utils.h"
#include "labels.h"
#include "xalloc.h"


// Local to global counter ID's per ptask 
local_hwc_data_t LocalHWCData = { NULL, 0 };

// Unique counters list 
global_hwc_data_t GlobalHWCData = { NULL, 0 };


/**
 * dump_counter_data
 *
 * Prints the contents of LocalHWCData and GlobalHWCData structures for debugging purposes.
 * The former maps the information of the local counter id's assigned to each thread in each 
 * ptask and their assigned global id, and the latter contains a list of all counters 
 * used in all ptasks and threads without repetitions and all their associated counter information.
 */
static void dump_counter_data()
{
        int i = 0, j = 0;

        for (i = 0; i < LocalHWCData.num_ptasks; i++)
        {
                fprintf(stderr, "[DEBUG] Dumping %d counter definitions for ptask %d\n", LocalHWCData.ptask_counters[i].num_counters, i+1);

                for (j=0; j<LocalHWCData.ptask_counters[i].num_counters; j++)
                {
                        hwc_id_t *local_to_global = &(LocalHWCData.ptask_counters[i].local_to_global[j]);

                        fprintf(stderr, "[DEBUG] Counter %d: ptask=%d local_id=%d global_id=%d\n", j, local_to_global->ptask, local_to_global->local_id, local_to_global->global_id);
                }
        }

	fprintf(stderr, "[DEBUG] Dumping unified counters information\n");
	for (i = 0; i < GlobalHWCData.num_counters; i++)
	{
		fprintf(stderr, "[DEBUG] Counter %d: name=%s description=%s global_id=%d used=%d\n", 
			i,
			GlobalHWCData.counters_info[i].name,	
			GlobalHWCData.counters_info[i].description,	
			GlobalHWCData.counters_info[i].global_id,	
			GlobalHWCData.counters_info[i].used);
	}
}


/**
 * Manipulate_HWC_Names
 *
 * Edits the given counter name in order to ignore parts of the name that may change but 
 * still refer to the same counter, so that we assign the same ID's to all variations.
 *
 * @param hwc_name Original counter name
 * @return Modified counter name
 */
static char * Manipulate_HWC_Names(char *hwc_name)
{
	char *hwc_modified_name = strdup(hwc_name);

	/* Infiniband counters look like: infiniband:::mlx5_0_1:port_xmit_data,
	 * where the 3 digits in mlx5_0_1 represent the driver version, NIC identifier, 
	 * and port identifier. The following patch overrides the driver version so that
	 * we assign the same ID's to the same counters for mlx4 and mlx5 versions.
	 */
	if ((strlen(hwc_name) > 17) && (!strncmp(hwc_name, "infiniband:::mlx", 16)))
	{
		hwc_modified_name[16] = '_';
	}

	return hwc_modified_name;
}


/**
 * HardwareCounters_AssignGlobalID
 *
 * Called for each H entry in the global SYM that declares a new counter for the given ptask.
 * Assigns an unique Paraver type for every possible counter seen across all ptasks.
 * Potential collisions are avoided by rehashing the counter name.
 *
 * @param ptask Application identifier
 * @param local_id Local counter identifier assigned during the tracing phase for the given ptask
 * @param definition String containing counter's short name and long description
 */
void HardwareCounters_AssignGlobalID (int ptask, int local_id, char *definition)
{
        int i = 0;
        int ntokens = 0;
        char **tokensCounterName = NULL;

        if (ptask > LocalHWCData.num_ptasks)
        {
                LocalHWCData.ptask_counters = xrealloc(LocalHWCData.ptask_counters, ptask * sizeof(ptask_hwc_t));
                for (i = LocalHWCData.num_ptasks; i<ptask; i++)
                {
                        // HardwareCounters_AssignGlobalID is called from ProcessArgs incrementally for every ptask,
                        // if one ptask is skipped, no global SYM file was provided for it, so the counters for this
                        // ptask won't appear in the PCF and the local codes won't be translated!
                        LocalHWCData.ptask_counters[i].local_to_global = NULL;
                        LocalHWCData.ptask_counters[i].num_counters = 0;
                }
                LocalHWCData.num_ptasks = ptask;
                fprintf(stderr, "mpi2prv: Retrieving hardware counters definitions for ptask %d from global SYM.\n", ptask);
        }

        // Fill counters information for current ptask
        ptask_hwc_t *ptask_counters = &(LocalHWCData.ptask_counters[ptask - 1]);
        ptask_counters->num_counters ++;
        ptask_counters->local_to_global = xrealloc(ptask_counters->local_to_global, (ptask_counters->num_counters) * sizeof(hwc_id_t));

        hwc_id_t *local_to_global = &(ptask_counters->local_to_global[ptask_counters->num_counters - 1]);

        local_to_global->local_id = local_id;
        local_to_global->ptask = ptask;

        ntokens = __Extrae_Utils_explode(definition, " ", &tokensCounterName);
        char *hwc_original_name = tokensCounterName[0];
        char *hwc_long_description = index(definition, '[');

	// Hack to ignore driver version in the counter name (for Infiniband counters, etc.)
	char *hwc_short_name = Manipulate_HWC_Names(hwc_original_name);

        // Compute global id for this counter and verify there's no colliding ID's with counters from any other ptask
        char rehash_trailer[9];
        xmemset(rehash_trailer, '\0', sizeof(rehash_trailer));

        int assigned = FALSE;
        int to_same_counter = FALSE;

        do
        {
                assigned = to_same_counter = FALSE;

                int len = (strlen(hwc_short_name) + strlen(rehash_trailer) + 1);
                char *hash_string = xmalloc_and_zero(len * sizeof(char));
                strcat(hash_string, hwc_short_name);
                strcat(hash_string, rehash_trailer);

                local_to_global->global_id = GET_PARAVER_CODE_FOR_HWC(local_id, hash_string);
                xfree(hash_string);

                // Sanity check this global_id is not assigned already to a counter with different name
                for (i=0; ((i<GlobalHWCData.num_counters) && (!assigned)); i++)
                {
                        if (GlobalHWCData.counters_info[i].global_id == local_to_global->global_id)
                        {
                                assigned = TRUE;

                                if (strcmp(GlobalHWCData.counters_info[i].name, hwc_original_name) == 0)
                                {
                                        to_same_counter = TRUE;
                                }
                        }
                }
                if ((assigned) && (!to_same_counter))
                {
                        // Append the global id to the counter name for the next rehash to avoid collisions and retry
                        snprintf(rehash_trailer, sizeof(rehash_trailer), "%d", local_to_global->global_id);
                        assigned = FALSE;
                }
                else if (!assigned)
                {
                        // Add new entry in GlobalHWCData list
                        int idx = GlobalHWCData.num_counters ++;

                        GlobalHWCData.counters_info = xrealloc(GlobalHWCData.counters_info, GlobalHWCData.num_counters * sizeof(hwc_info_t));
                        GlobalHWCData.counters_info[idx].name = strdup(hwc_original_name);
                        GlobalHWCData.counters_info[idx].description = strdup(hwc_long_description);
                        GlobalHWCData.counters_info[idx].global_id = local_to_global->global_id;
                        GlobalHWCData.counters_info[idx].used = FALSE;
                        assigned = TRUE;
                }
        }
        while (!assigned);

        if (strlen(rehash_trailer) > 0)
        {
                // There was a collision in the counters identifiers
                fprintf(stderr, "mpi2prv: WARNING: Local ID %d for hardware counter %s from ptask %d collided with another counter. This was rehashed into ID %d only for this trace.\n",
                        local_to_global->local_id,
                        hwc_original_name,
                        local_to_global->ptask,
                        local_to_global->global_id);
        }

        for (i=0; i<ntokens; i++) xfree(tokensCounterName[i]);
        xfree(tokensCounterName);
	xfree(hwc_short_name);
}


/**
 * HardwareCounters_LocalToGlobalID
 *
 * Translate the local counter id 'local_id' for the given 'ptask' into its univocal 'global_id' code.
 *
 * @param ptask Application identifier
 * @param local_id Local counter identifier assigned during the tracing phase for the given ptask
 * @return Unified global counter identifier without collisions
 */
int HardwareCounters_LocalToGlobalID(int ptask, int local_id)
{
        int i = 0;

        if (ptask > 0 && ptask <= LocalHWCData.num_ptasks)
        {
                int idx = ptask - 1;

                for (i = 0; i < LocalHWCData.ptask_counters[idx].num_counters; i ++)
                {
                        hwc_id_t *local_to_global = &(LocalHWCData.ptask_counters[idx].local_to_global[i]);

                        if (local_to_global->local_id == local_id)
                        {
                                return local_to_global->global_id;
                        }
                }
        }
        fprintf(stderr, "mpi2prv: WARNING: Could not find global HWC identifier for ptask=%d local_id=%d. Did you pass the SYM file to the merger?\n", ptask, local_id);

        /* When global SYM file is not provided, we don't have local to global translations, so we need
         * to keep the local identifier. However, the local identifier can be negative in some cases 
         * (e.g. PAPI presets id for TOT_INS 0x80000032 is larger than max integer representation and turns on the negative bit),
         * thus we apply our legacy transformation of moving the numbers to the range 4X million
         */  
        return LEGACY_HWC_COUNTER_TYPE(local_id);
}


/**
 * mark_counter_used
 *
 * Marks the counter identified by the given global_id as used (any thread has read this counter at least once)
 *
 * @param global_id Global counter identifier
 */
static void mark_counter_used(int global_id)
{
        int i;

        for (i=0; i<GlobalHWCData.num_counters; i++)
        {
                if (GlobalHWCData.counters_info[i].global_id == global_id)
                {
                        GlobalHWCData.counters_info[i].used = TRUE;
                        return;
                }
        }
}


/**
 * HardwareCounters_GetUsed
 *
 * Returns the sublist of unique counters that have been measured during the tracing.
 * 
 * @param used_counters_io List of unique counters that were used (I/O)
 * @return Count of used counters
 */
int HardwareCounters_GetUsed(hwc_info_t ***used_counters_io)
{
	int i = 0, idx = 0, num_used = 0;
	hwc_info_t **used_counters = NULL;

	for (i = 0; i < GlobalHWCData.num_counters; i++)
	{
		if (GlobalHWCData.counters_info[i].used) 
		{
			num_used ++;
		}
	}

	if (num_used > 0)
	{
		used_counters = xmalloc(num_used * sizeof(hwc_info_t *));

		i = 0;
		while (idx < num_used)
		{
			if (GlobalHWCData.counters_info[i].used)
			{
				used_counters[idx] = &(GlobalHWCData.counters_info[i]);
				idx ++;
			}
			i ++;
		}
	}
	*used_counters_io = used_counters;
	return num_used;
}


/**
 * get_set_ids
 *
 * Retrieve from the given 'thread' data the array of counter identifiers (local & global) for the given 'set_id'
 *
 * @param ptask Application identifier
 * @param task Task identifier
 * @param thread Thread identifier
 * @param set_id Counter set identifier
 *
 * @return Array of translations from local to global identifiers for the counters belonging to set 'set_id'
 */
static hwc_id_t * get_set_ids(int ptask, int task, int thread, int set_id)
{
	thread_t *Sthread = ObjectTree_getThreadInfo(ptask, task, thread);

	if ((set_id+1 > Sthread->num_HWCSets) || (set_id < 0))
	{
		fprintf(stderr, "mpi2prv: WARNING: Could not find definitions for HWC set '%d' for object (%d.%d.%d). "
				"Counters readings for this set will not appear in the final trace!\n", set_id, ptask, task, thread);
		return NULL;
	}
	else
	{
		return Sthread->HWCSets[set_id];
	}
}


/**
 * HardwareCounters_NewSetDefinition
 *
 * Called once per HWC_DEF_EV event emitted from the tracing. These events appear at the beginning of 
 * each mpit, and are emitted with incremental set id's one after the other. The merger sees
 * the events in the same order, so this function receives 'newSet' incrementing by 1 in each
 * invocation for the given 'ptask, task, thread'.
 *
 * @param ptask Application identifier
 * @param task Task identifier
 * @param thread Thread identifier
 * @param newSet Counter set identifier
 * @param HWCIds Array of local counter identifiers
 */
void HardwareCounters_NewSetDefinition (int ptask, int task, int thread, int newSet, long long *HWCIds)
{
	thread_t *Sthread;

	Sthread = ObjectTree_getThreadInfo(ptask, task, thread);

	if (newSet >= Sthread->num_HWCSets)
	{
		int i, j;

		Sthread->HWCSets = xrealloc(Sthread->HWCSets, (newSet+1) * sizeof(hwc_id_t *));
		Sthread->HWCSets[newSet] = xmalloc(MAX_HWC * sizeof(hwc_id_t));

		/*
		 * Initialize all counters for the new set as unused, and do the 
		 * same for all sets between the last set definition seen by this 
		 * thread up to the current new set. Since the sets definitions come
		 * in order, this will iterate only once for the new set, but we
		 * leave this sanity check just in case the set definitions could 
		 * come unsorted in the future.
		 */
		for (i = Sthread->num_HWCSets; i <= newSet; i++)
		{
			for (j=0; j<MAX_HWC; j++)
			{
				Sthread->HWCSets[i][j].local_id = Sthread->HWCSets[i][j].global_id = NO_COUNTER;
			}
		}

		/* 
		 * Save the local and global hwc id's for the current new set. 
		 */
		if (HWCIds != NULL)
		{
			for (i = 0; i < MAX_HWC; i++)
			{
				if (HWCIds[i] != NO_COUNTER)
				{
					Sthread->HWCSets[newSet][i].ptask = ptask;
					Sthread->HWCSets[newSet][i].local_id = (int)HWCIds[i];
					Sthread->HWCSets[newSet][i].global_id = HardwareCounters_LocalToGlobalID(ptask, HWCIds[i]);
				}
			}
		}
		Sthread->num_HWCSets = newSet + 1;
	}
}


/**
 * HardwareCounters_Change
 *
 * Called when the merger sees a HWC_CHANGE_EV.
 * Returns two arrays of types (with the global HWC id) and values set to 0, and 
 * a counter for the number of valid counters in the given set 'newSetId'. 
 * The first position of the arrays 'outtypes' and 'outvalues' contain the 
 * special event HWC_GROUP_ID whose value indicates the active set.
 * Remaining positions contain those counters in set 'newSetId' that did
 * not appear in the previous set for this (ptask, task, thread).
 *
 * @param ptask Application identifier
 * @param task Task identifier
 * @param thread Thread identifier
 * @param change_time Timestamp of the current counter set change
 * @param newSetId Counter set identifier in use after the current counter set change
 * @param outtypes Array of global counter identifiers (Paraver types). 1st position is a HWC_GROUP_ID event, this event marks the set change (I/O)
 * @param outvalues Array of counter values initialized to 0 at the time of the change. 1st position is the new set identifier (I/O)
 *
 * @return Number of counters in the new set + 1 (1st event marking the set change; matches the size of outtypes and outvalues)
 */
int HardwareCounters_Change (int ptask, int task, int thread, unsigned long long change_time, int newSetId, unsigned int *outtypes, unsigned long long *outvalues)
{
	int i = 0;
	thread_t *Sthread = ObjectTree_getThreadInfo(ptask, task, thread);

	// Count how many set changes has seen this thread
	int first_hwc_change = (Sthread->HWCChange_count == 0);
	Sthread->HWCChange_count++;

	// Save the last change timestamp
	Sthread->last_hw_group_change = change_time;

	// Retrieve ids for the exiting set 
	hwc_id_t *oldSetHWCIds = get_set_ids (ptask, task, thread, Sthread->current_HWCSet);

	// Update the current counters set
	Sthread->current_HWCSet = newSetId;

	// Mark the new active set in the first output event
	int outcount = 0;
	outtypes[0] = HWC_GROUP_ID;
	outvalues[0] = newSetId + 1;
	outcount ++;

	hwc_id_t *newSetHWCIds = get_set_ids (ptask, task, thread, newSetId);
	if (newSetHWCIds != NULL)
	{
		for (i = 0; i < MAX_HWC; i++)
		{	
			int present_in_old_set = FALSE;
			
			if (oldSetHWCIds != NULL)
			{
				int j;

				for (j = 0; (j < MAX_HWC) && (!present_in_old_set); j ++)
				{
					if (newSetHWCIds[i].global_id == oldSetHWCIds[j].global_id)
					{
						present_in_old_set = TRUE;
					}
				}
			}

			// Emit zero readings only for counters in the new set that did not appear in the previous set.
			// Always emit zero the first time the counters are changed.
			if ((newSetHWCIds[i].local_id != NO_COUNTER) && ((!present_in_old_set) || (first_hwc_change)))
			{
				outtypes[outcount] = newSetHWCIds[i].global_id;
				outvalues[outcount] = 0;

				mark_counter_used( outtypes[outcount] );

				outcount ++;
			}
		}
	}

	return outcount; 
}


/**
 * HardwareCounters_SetOverflow
 *
 * Marks in the given thread data which counters are working in overflow mode
 *
 * @param ptask Application identifier
 * @param task Task identifier
 * @param thread Thread identifier
 * @param Event Record from a HWC_SET_OVERFLOW_EV event that marks in HWCValues which counters are working in overflow mode (marked as SAMPLE_COUNTER)
 */
void HardwareCounters_SetOverflow (int ptask, int task, int thread, event_t *Event)
{
	int cnt;
	thread_t *Sthread = ObjectTree_getThreadInfo(ptask, task, thread);
	int set_id = Sthread->current_HWCSet;

	for (cnt = 0; cnt < MAX_HWC; cnt++)
		if (Event->HWCValues[cnt] == SAMPLE_COUNTER)
			Sthread->HWCSets[set_id][cnt].local_id = SAMPLE_COUNTER;
}


/**
 * HardwareCounters_Emit
 *
 * Called for each event in the mpits that has counters measuremens. Since the event record only contains the values read, 
 * the types are retrieved from the (ptask, task, thread) in the object tree. The proper types are pointed by the field
 * current_HWCSet, which is updated when there's an event of hardware counters change (see HardwareCounters_Change).
 *
 * @param ptask Application identifier
 * @param task Task identifier
 * @param thread Thread identifier
 * @param time Timestamp of the given Event 
 * @param Event The event record with counters measured
 * @param outtype Output array of global counter identifiers (Paraver type)
 * @param outvalue Output array of counter values
 * @param absolute Set to 1 if we want absolute values; 0 to reset after each measurement
 *  
 * @return Number of counter values to emit 
 */
int HardwareCounters_Emit (int ptask, int task, int thread,
	unsigned long long time, event_t * Event, unsigned int *outtype,
	unsigned long long *outvalue, int absolute)
{
	int cnt = 0;
	int outcount = 0;
	thread_t *Sthread = ObjectTree_getThreadInfo(ptask, task, thread);

	hwc_id_t *SetHWCIds = get_set_ids (ptask, task, thread, Sthread->current_HWCSet);

	/* Don't emit hwc that coincide in time  with a hardware counter group change.
	 * Special treatment for the first HWC change, which must be excluded in order
	 * to get the first counters (which shall be 0).
	 * However, we must track the value of counters if SAMPLING_SUPPORT */
#if defined(PAPI_COUNTERS) && defined (SAMPLING_SUPPORT)
	if (Sthread->last_hw_group_change == time)
	{
		for (cnt = 0; cnt < MAX_HWC; cnt++)
		{
			if (SetHWCIds[cnt].local_id != NO_COUNTER &&
                            SetHWCIds[cnt].local_id != SAMPLE_COUNTER)
                        {
				if (Sthread->HWCChange_count == 1)
				{
	                                if (!absolute)
	                                {
	                                        outvalue[outcount] = 0;
	                                        outtype[outcount]  = SetHWCIds[cnt].global_id;
	                                }
	                                else
	                                {
	                                        outvalue[outcount] = 0;
	                                        outtype[outcount]  = SetHWCIds[cnt].global_id + HWC_DELTA_ABSOLUTE;
	                                }
	                                Sthread->counters[cnt] = 0; 
	                                outcount ++;
				}
				else Sthread->counters[cnt] = Event->HWCValues[cnt];
			}
		}
		return outcount;
	}
#endif

#if defined(PAPI_COUNTERS) || defined(PMAPI_COUNTERS)
	for (cnt = 0; cnt < MAX_HWC; cnt++)
	{
		/* If using PAPI, they can be stored in absolute or relative manner,
		 * depending whether sampling was activated or not 
		 */
# if defined(SAMPLING_SUPPORT)
		if (SetHWCIds[cnt].local_id != NO_COUNTER && SetHWCIds[cnt].local_id != SAMPLE_COUNTER)
# else 
		if (SetHWCIds[cnt].local_id != NO_COUNTER)
# endif
		{
# if defined(SAMPLING_SUPPORT)
			// Protect when counters are incorrect (major timestamp, lower counter value)
			if (Event->HWCValues[cnt] >= Sthread->counters[cnt])
# endif
			{
				if (!absolute) // Want to emit relative values
				{
					unsigned long long substract_previous = 0;

# if defined(SAMPLING_SUPPORT) || defined(PMAPI_COUNTERS)
					// Substract previous counters reading manually when sampling enabled or values come from PMAPI as they are never reset
					substract_previous = Sthread->counters[cnt];
# endif
					outvalue[outcount] = Event->HWCValues[cnt] - substract_previous;
					outtype[outcount]  = SetHWCIds[cnt].global_id;
				}
				else // Want to emit absolute values
				{
					outvalue[outcount] = Event->HWCValues[cnt];
					outtype[outcount]  = SetHWCIds[cnt].global_id + HWC_DELTA_ABSOLUTE;
				}
				outcount ++;
			}

			// Save current values to substract in next emission that requires manual reset
			Sthread->counters[cnt] = Event->HWCValues[cnt];
		}
	}
#endif
	return outcount;
}


/**
 * HardwareCounters_Show
 *
 * Dumps the counters values from the given event record.
 *
 * @param Event The event record with counters measured
 * @param ncounters Number of counter values to dump 
 */
void HardwareCounters_Show (const event_t * Event, int ncounters)
{
  int cnt;
  fprintf (stdout, "COUNTERS: ");
  for (cnt = 0; cnt < ncounters; cnt++)
    fprintf (stdout, "[%lld] ", Event->HWCValues[cnt]);
  fprintf (stdout, "\n");
}


#if defined(PAPI_COUNTERS)

#include <perfmon/pfmlib.h>
#include <string.h>


/**
 * check_if_uncore_in_PFM
 *
 * Query libpfm to see if the given counter is an uncore counter or not.
 *
 * @param event_name Short name of the given counter
 *
 * @return 1 if is uncore counter; 0 otherwise
 */
int check_if_uncore_in_PFM(char *event_name)
{
	static int pfm_initialized = 0;
	pfm_err_t ret = 0;

	if (event_name == NULL) return 0;

	if (!pfm_initialized)
	{
		if ((ret = pfm_initialize()) != PFM_SUCCESS)
		{
			fprintf(stderr, "ERROR: pfm_initialize failed: %s\n", pfm_strerror(ret));
		}
		else pfm_initialized = 1;
	}

	if (pfm_initialized)
	{
		pfm_pmu_encode_arg_t ev;
		xmemset(&ev, 0, sizeof(ev));

		pfm_event_info_t info;
		xmemset(&info, 0, sizeof(info));

		pfm_pmu_info_t pinfo;
		xmemset(&pinfo, 0, sizeof(pinfo));

                // Remove the :cpu= attribute, can't query pfm otherwise
		char *cputag = NULL;
		char *event_name_without_cpu_attr = strdup(event_name);
                if ((cputag = strstr(event_name_without_cpu_attr, ":cpu=")) != NULL)
                {
                        cputag[0] = '\0';
                }

		//fprintf(stderr, "[DEBUG] pfm_get_os_event_encoding %s\n", event_name_without_cpu_attr);
		pfm_get_os_event_encoding(event_name_without_cpu_attr, PFM_PLM0|PFM_PLM3, PFM_OS_NONE, &ev);
		pfm_get_event_info(ev.idx, PFM_OS_NONE, &info);
		pfm_get_pmu_info(info.pmu, &pinfo);

		xfree(event_name_without_cpu_attr);

		/*
		 * Possible values for pinfo.type:
		 * PFM_PMU_TYPE_UNKNOWN=0  // unknown PMU type
                 * PFM_PMU_TYPE_CORE       // processor core PMU
                 * PFM_PMU_TYPE_UNCORE     // processor socket-level PMU
                 * PFM_PMU_TYPE_OS_GENERIC // generic OS-provided PMU
		 */
		//fprintf(stderr, "[DEBUG] check_if_uncore_in_PFM returns %d\n", (pinfo.type == PFM_PMU_TYPE_UNCORE));
		return (pinfo.type == PFM_PMU_TYPE_UNCORE);
	}
	//fprintf(stderr, "[DEBUG] check_if_uncore_in_PFM returns FALSE\n");
	return FALSE;
}

#endif /* PAPI_COUNTERS */


#if defined(PARALLEL_MERGE)

/**
 * Share_HWC_Before_Processing_MPITS
 *
 * Broadcast LocalHWCData and GlobalHWCData structures from master to all worker processes. 
 * This is necessary in order to have all parallel merger processes have the same counter identifiers,
 * as only the master rank parses the *.SYM files. This is done at the beginning of the merging
 * phase, once the *.SYM files have been parsed.
 *
 * @param rank The parallel merger process identifier.
 */
void Share_HWC_Before_Processing_MPITS (int rank)
{
        int position = 0;
        void *buffer = NULL;
        int buffer_size = 0;
        int i = 0, j = 0, size = 0;

        if (rank == 0)
        {
        	// Serialize LocalHWCData structure
                MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &size);
                resize(buffer, size);
                MPI_Pack(&LocalHWCData.num_ptasks, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);

                for (i = 0; i < LocalHWCData.num_ptasks; i++)
                {
                        ptask_hwc_t *ptask_counters = &(LocalHWCData.ptask_counters[i]);

                        MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &size);
                        resize(buffer, size);
                        MPI_Pack(&(ptask_counters->num_counters), 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);

                        for (j = 0; j < ptask_counters->num_counters; j++)
                        {
                                hwc_id_t *local_to_global = &(ptask_counters->local_to_global[j]);

                                MPI_Pack_size(3, MPI_INT, MPI_COMM_WORLD, &size);
                                resize(buffer, size);
                                MPI_Pack(&local_to_global->ptask, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);
                                MPI_Pack(&local_to_global->local_id, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);
                                MPI_Pack(&local_to_global->global_id, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);
                        }
                }

		// Serialize GlobalHWCData structure
                MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &size);
                resize(buffer, size);
                MPI_Pack(&GlobalHWCData.num_counters, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);

                for (i = 0; i < GlobalHWCData.num_counters; i++)
                {
                        int name_len = 0, description_len = 0;
                        hwc_info_t *info = &(GlobalHWCData.counters_info[i]);

                        MPI_Pack_size(4, MPI_INT, MPI_COMM_WORLD, &size);
                        resize(buffer, size);
                        MPI_Pack(&info->global_id, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);
                        MPI_Pack(&info->used, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);
                        name_len = strlen(info->name);
                        MPI_Pack(&name_len, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);
                        description_len = strlen(info->description);
                        MPI_Pack(&description_len, 1, MPI_INT, buffer, buffer_size, &position, MPI_COMM_WORLD);

                        MPI_Pack_size(name_len, MPI_CHAR, MPI_COMM_WORLD, &size);
                        resize(buffer, size);
                        MPI_Pack(info->name, name_len, MPI_CHAR, buffer, buffer_size, &position, MPI_COMM_WORLD);

                        MPI_Pack_size(description_len, MPI_CHAR, MPI_COMM_WORLD, &size);
                        resize(buffer, size);
                        MPI_Pack(info->description, description_len, MPI_CHAR, buffer, buffer_size, &position, MPI_COMM_WORLD);
                }
        }

        // Root broadcasts the serialized structures
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0)
        {
                buffer = xmalloc(buffer_size);
        }
        MPI_Bcast(buffer, buffer_size, MPI_PACKED, 0, MPI_COMM_WORLD);

        // Workers unpack the structures
        if (rank != 0)
        {
                position = 0;

                MPI_Unpack(buffer, buffer_size, &position, &(LocalHWCData.num_ptasks), 1, MPI_INT, MPI_COMM_WORLD);

                LocalHWCData.ptask_counters = xmalloc(LocalHWCData.num_ptasks * sizeof(ptask_hwc_t));
                for (i = 0; i < LocalHWCData.num_ptasks; i++)
                {
                        ptask_hwc_t *ptask_counters = &(LocalHWCData.ptask_counters[i]);

                        MPI_Unpack(buffer, buffer_size, &position, &(ptask_counters->num_counters), 1, MPI_INT, MPI_COMM_WORLD);

                        ptask_counters->local_to_global = xmalloc(ptask_counters->num_counters * sizeof(hwc_id_t));
                        for (j = 0; j < ptask_counters->num_counters; j++)
                        {
                                hwc_id_t *local_to_global = &(ptask_counters->local_to_global[j]);

                                MPI_Unpack(buffer, buffer_size, &position, &(local_to_global->ptask), 1, MPI_INT, MPI_COMM_WORLD);
                                MPI_Unpack(buffer, buffer_size, &position, &(local_to_global->local_id), 1, MPI_INT, MPI_COMM_WORLD);
                                MPI_Unpack(buffer, buffer_size, &position, &(local_to_global->global_id), 1, MPI_INT, MPI_COMM_WORLD);
                        }
                }

                MPI_Unpack(buffer, buffer_size, &position, &(GlobalHWCData.num_counters), 1, MPI_INT, MPI_COMM_WORLD);

                GlobalHWCData.counters_info = xmalloc(GlobalHWCData.num_counters * sizeof(hwc_info_t));
                for (i = 0; i < GlobalHWCData.num_counters; i++)
                {
                        int name_len = 0, description_len = 0;
                        hwc_info_t *info = &(GlobalHWCData.counters_info[i]);

                        MPI_Unpack(buffer, buffer_size, &position, &(info->global_id), 1, MPI_INT, MPI_COMM_WORLD);
                        MPI_Unpack(buffer, buffer_size, &position, &(info->used), 1, MPI_INT, MPI_COMM_WORLD);
                        MPI_Unpack(buffer, buffer_size, &position, &(name_len), 1, MPI_INT, MPI_COMM_WORLD);
                        MPI_Unpack(buffer, buffer_size, &position, &(description_len), 1, MPI_INT, MPI_COMM_WORLD);

                        info->name = xmalloc_and_zero((name_len + 1) * sizeof(char));
                        MPI_Unpack(buffer, buffer_size, &position, info->name, name_len, MPI_CHAR, MPI_COMM_WORLD);

                        info->description = xmalloc_and_zero((description_len + 1) * sizeof(char));
                        MPI_Unpack(buffer, buffer_size, &position, info->description, description_len, MPI_CHAR, MPI_COMM_WORLD);
                }
        }
}


/**
 * Share_HWC_After_Processing_MPITS
 *
 * Gather counter usage information from all worker processes into the master in parallel merge. 
 * This is done after all the *.mpit files have been processed, and is necessary so as the master 
 * has the full list of used global counters to write entries in the PCF only for those used.
 *
 * @param rank The parallel merger process identifier.
 */
void Share_HWC_After_Processing_MPITS (int rank)
{
        int i = 0;
        int *counters_used = xmalloc(GlobalHWCData.num_counters * sizeof(int));
        int *used_reduction = NULL;

        if (rank == 0) used_reduction = xmalloc_and_zero(GlobalHWCData.num_counters * sizeof(int));

        for (i = 0; i < GlobalHWCData.num_counters; i ++)
        {
                counters_used[i] = GlobalHWCData.counters_info[i].used;
        }

        MPI_Reduce(counters_used, used_reduction, GlobalHWCData.num_counters, MPI_INT, MPI_LOR, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
                for (i = 0; i < GlobalHWCData.num_counters; i ++)
                {
                        GlobalHWCData.counters_info[i].used = used_reduction[i];
                }
                xfree(used_reduction);
        }
        xfree(counters_used);
}

#endif /* PARALLEL_MERGE */

#endif /* USE_HARDWARE_COUNTERS  || HETEROGENEOUS_SUPPORT */

