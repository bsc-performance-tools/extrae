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
#include "pmapi_hwc.h"
#include "hwc_version.h"

/*------------------------------------------------ Static Variables ---------*/

static pm_info2_t ProcessorMetric_Info; /* On AIX pre 5.3 it was pm_info_t */
static pm_groups_info_t HWCGroup_Info;

int MAX_HWC_reported_by_PMAPI = 0;

int HWCBE_PMAPI_Add_Set (int pretended_set, int rank, int ncounters, char **counters,
	char *domain, char *change_at_globalops, char *change_at_time, 
	int num_overflows, char **overflow_counters, unsigned long long *overflow_values)
{
	int i, num_set = HWC_num_sets;
	int GROUP;
	char *strtoul_check;
	char *counter_last_position = &(counters[0][strlen(counters[0])]);
	
	if (ncounters == 0 || counters == NULL)
		return 0;

	if (ncounters != 1)
	{
		fprintf (stderr, PACKAGE_NAME": PMAPI layer just supports 1 HWC group per set (see set %d)\n", pretended_set);
	}
	
	HWC_sets = (struct HWC_Set_t *) realloc (HWC_sets, sizeof(struct HWC_Set_t)* (HWC_num_sets+1));
	if (HWC_sets == NULL)
	{
		fprintf (stderr, PACKAGE_NAME": Cannot allocate memory for HWC_set (rank %d)\n", rank);
		return 0;
	}

	/* Initialize this set */
	HWC_sets[num_set].pmprog.mode.w = 0;
	HWC_sets[num_set].pmprog.mode.b.is_group = 1;
	HWC_sets[num_set].num_counters = ProcessorMetric_Info.maxpmcs;

	/* counter_last_position will hold the address of the end of the 
	   counter[i] string 
	   This shall be compared with strtoul_check to know if the hex
	   is correct or not
	*/

	GROUP = strtoul (counters[0], &strtoul_check, 10);

	if (strtoul_check != counter_last_position)
	{
		/* Not numerical counter? check if it's on the short names table */
		int i, found;
		for (i = 0, found = FALSE; i < HWCGroup_Info.maxgroups; i++)
		{
			found = (strcmp (counters[0], HWCGroup_Info.event_groups[GROUP].short_name) == 0);
			if (found)
			{
				GROUP = i;
				break;
			}
		}
		if (!found)
		{
			fprintf (stderr, PACKAGE_NAME": Cannot parse HWC %s in set %d, skipping\n", counters[0], pretended_set);
			return 0;
		}
	}
	
	if (HWC_sets[num_set].num_counters == 0)
	{
		if (rank == 0)
			fprintf (stderr, PACKAGE_NAME": Set %d of counters seems to be empty/invalid, skipping\n", pretended_set);
		return 0;
	}

	if (GROUP >= HWCGroup_Info.maxgroups)
	{
		if (rank == 0)
			fprintf (stderr, PACKAGE_NAME": Error! Group %d is beyond the maximum number of groups available (0 to %d). Check set %d.\n", GROUP, HWCGroup_Info.maxgroups-1, pretended_set);
		return 0;
	}

	/* Copy counters from group to our buffer */
	HWC_sets[num_set].group = HWC_sets[num_set].pmprog.events[0] = GROUP;
	for (i = 0; i < HWC_sets[num_set].num_counters; i++)
		HWC_sets[num_set].counters[i] = HWCGroup_Info.event_groups[GROUP].events[i];

	/* Just check if the user wants us to change the counters in some manner */
	if (change_at_time != NULL)
	{
		HWC_sets[num_set].change_at = getTimeFromStr (change_at_time, 
			"change-at-time", rank);
		HWC_sets[num_set].change_type = 
				(HWC_sets[num_set].change_at == 0)?CHANGE_NEVER:CHANGE_TIME;
	}
	else if (change_at_globalops != NULL)
	{
		HWC_sets[num_set].change_at = strtoul (change_at_globalops, (char **) NULL, 10);
		HWC_sets[num_set].change_type = 
			(HWC_sets[num_set].change_at == 0)?CHANGE_NEVER:CHANGE_GLOPS;
	}
	else
		HWC_sets[num_set].change_type = CHANGE_NEVER;

	if (domain != NULL)
	{
		if (!strcasecmp(domain, "all"))
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PMAPI domain set to ALL for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].pmprog.mode.b.kernel = 1;
			HWC_sets[num_set].pmprog.mode.b.user = 1;
			HWC_sets[num_set].pmprog.mode.b.hypervisor = 1;
			HWC_sets[num_set].pmprog.mode.b.count = 1;
		}	
		else if (!strcasecmp(domain, "kernel"))
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PMAPI domain set to KERNEL for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].pmprog.mode.b.kernel = 1;
			HWC_sets[num_set].pmprog.mode.b.user = 0;
			HWC_sets[num_set].pmprog.mode.b.hypervisor = 1;
			HWC_sets[num_set].pmprog.mode.b.count = 1;
		}	
		else if (!strcasecmp(domain, "user"))
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PMAPI domain set to USER for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].pmprog.mode.b.kernel = 0;
			HWC_sets[num_set].pmprog.mode.b.user = 1;
			HWC_sets[num_set].pmprog.mode.b.hypervisor = 0;
			HWC_sets[num_set].pmprog.mode.b.count = 1;
		}	
		else if (!strcasecmp(domain, "other"))
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PMAPI domain set to OTHER for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].pmprog.mode.b.kernel = 0;
			HWC_sets[num_set].pmprog.mode.b.user = 0;
			HWC_sets[num_set].pmprog.mode.b.hypervisor = 1;
			HWC_sets[num_set].pmprog.mode.b.count = 1;
		}	
		else
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PMAPI domain set to USER for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].pmprog.mode.b.kernel = 0;
			HWC_sets[num_set].pmprog.mode.b.user = 1;
			HWC_sets[num_set].pmprog.mode.b.hypervisor = 0;
			HWC_sets[num_set].pmprog.mode.b.count = 1;
		}
	} /* domain != NULL */
	else
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": PMAPI domain set to USER for HWC set %d\n",
				pretended_set);
		HWC_sets[num_set].pmprog.mode.b.kernel = 0;
		HWC_sets[num_set].pmprog.mode.b.user = 1;
		HWC_sets[num_set].pmprog.mode.b.hypervisor = 0;
		HWC_sets[num_set].pmprog.mode.b.count = 1;
	}

	/* We validate this set */
	HWC_num_sets++;

	if (rank == 0)
	{
		fprintf (stdout, PACKAGE_NAME": HWC set %d refers to group %s (%d) which contains following counters < ", pretended_set, HWCGroup_Info.event_groups[GROUP].short_name, GROUP);
		for (i = 0; i < HWC_sets[num_set].num_counters; i++)
		{
			pm_events2_t *evp = NULL;
			int j, event = HWCGroup_Info.event_groups[GROUP].events[i];
			if (event != COUNT_NOTHING && ProcessorMetric_Info.maxevents[i] != 0)
			{
				/* find pointer to the event */
				for (j = 0; j < ProcessorMetric_Info.maxevents[i]; j++)
				{ 
					evp = ProcessorMetric_Info.list_events[i]+j;  
					if (event == evp->event_id)
						break;    
				}
				if (evp != NULL)
					printf("%s (0x%08x) ", evp->short_name, event);
			}
		} /* for (counter = 0; ... */
			
		fprintf (stdout, ">");

		if (HWC_sets[num_set].change_type == CHANGE_TIME)
			fprintf (stdout, " - changing every %lld nanoseconds\n", HWC_sets[num_set].change_at);
		else if (HWC_sets[num_set].change_type == CHANGE_GLOPS)
			fprintf (stdout, " - changing every %lld global operations\n", HWC_sets[num_set].change_at);
		else
			fprintf (stdout, " - never changes\n");

		fflush (stdout);
	}

	return HWC_sets[num_set].num_counters;
}

int HWCBE_PMAPI_Start_Set (UINT64 countglops, UINT64 time, int numset, int threadid)
{
	int rc;

	/* The given set is a valid one? */
	if (numset < 0 || numset >= HWC_num_sets)
		return FALSE;

	HWC_current_changeat = HWC_sets[numset].change_at;
	HWC_current_changetype = HWC_sets[numset].change_type;
	HWC_current_timebegin[threadid] = time;
	HWC_current_glopsbegin[threadid] = countglops;

	rc = pm_set_program_mythread (&(HWC_sets[numset].pmprog));
	if (rc != 0)
	{
		pm_error ("pm_set_program_mythread", rc);
		return FALSE;
	}
	else
	{
		TRACE_EVENT (time, HWC_CHANGE_EV, numset);
	}

	rc = pm_start_mythread ();
	if (rc != 0)
	{
		pm_error ("pm_start_mythread", rc);
		return FALSE;
	}

	return TRUE;
}

int HWCBE_PMAPI_Stop_Set (UINT64 time, int numset, int threadid)
{
	int rc;

	if (numset < 0 || numset >= HWC_num_sets)
		return FALSE;

	rc = pm_stop_mythread ();
	if (rc != 0)
	{
		pm_error ("pm_stop_mythread", rc);
		return FALSE;
	}

	rc = pm_delete_program_mythread ();
	if (rc != 0)
	{
		pm_error ("pm_delete_program_mythread", rc);
		return FALSE;
	}

	return TRUE;
}

void HWCBE_PMAPI_CleanUp (unsigned nthreads)
{
	UNREFERENCED_PARAMETER(nthreads);

	xfree (HWC_sets);
}

HWC_Definition_t *HWCBE_PMAPI_GetCounterDefinitions(unsigned *count)
{
	/* This is currently unimplemented */
	*count = 0;
	return NULL;
}

/******************************************************************************
 **      Function name : PMAPI_Initialize
 **
 **      Description :
 ******************************************************************************/

void HWCBE_PMAPI_Initialize (int TRCOptions)
{
	int rc;

	rc = pm_initialize (PM_VERIFIED|PM_UNVERIFIED|PM_CAVEAT|PM_GET_GROUPS, &ProcessorMetric_Info, &HWCGroup_Info, PM_CURRENT);
	if (rc != 0)
		pm_error ("pm_initialize", rc);

	MAX_HWC_reported_by_PMAPI = ProcessorMetric_Info.maxpmcs;

	fprintf (stdout, PACKAGE_NAME": PMAPI successfully initialized for %s processor with %d available PMCs.\n", ProcessorMetric_Info.proc_name, ProcessorMetric_Info.maxpmcs);
}

int HWCBE_PMAPI_Init_Thread (UINT64 time, int threadid, int forked)
{
	UNREFERENCED_PARAMETER(forked);

	HWC_Thread_Initialized[threadid] = HWCBE_PMAPI_Start_Set (0, time, HWC_current_set[threadid], threadid);
	return HWC_Thread_Initialized[threadid];
}

int HWCBE_PMAPI_Read (unsigned int tid, long long *store_buffer)
{
	pm_data_t counters;
	int i, num_hwc = MIN(MAX_HWC_reported_by_PMAPI, MAX_HWC);
	int rc;

	rc = pm_get_data_mythread (&counters);
	if (rc != 0)
	{
		fprintf (stderr, PACKAGE_NAME": pm_get_data_mythread failed for thread %d (%s:%d)\n",
			tid, __FILE__, __LINE__);
	}
	for (i = 0; i < num_hwc; i++)
		store_buffer[i] = counters.accu[i];
	for (i = num_hwc; i < MAX_HWC; i++)
		store_buffer[i] = 0;

	return (rc == 0);
}

