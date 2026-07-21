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
#include "xalloc.h"
#include "events.h"
#include "clock.h"
#include "threadid.h"
#include "record.h"
#include "trace_macros.h"
#include "wrapper.h"
#include "stdio.h"
#include "xml-parse.h"
#include "common_hwc.h"
#include "papi_hwc.h"
#include "hwc_version.h"
#include "papi.h"

#if defined(IS_BGL_MACHINE)
# define COUNTERS_INFO
# ifdef HAVE_BGL_PERFCTR_H
#  include <bgl_perfctr.h>
# endif
# ifdef HAVE_BGL_PERFCTR_EVENTS_H
#  include <bgl_perfctr_events.h>
# endif
#endif
#if defined(ENABLE_PEBS_SAMPLING)                                               
# include "sampling-intel-pebs.h"                                               
#endif                                                                          


/*------------------------------------------------ Static Variables ---------*/

static HWC_Definition_t *hwc_used = NULL;
static unsigned num_hwc_used = 0;

static unsigned topdown_level_global = TOPDOWN_DISABLED;
static unsigned topdown_num_counters = 0;

static const char *top_down_l1[TOPDOWN_NUM_COUNTERS_LVL1] = {
	"TOPDOWN_RETIRING_PERC", "TOPDOWN_BAD_SPEC_PERC",
	"TOPDOWN_FE_BOUND_PERC", "TOPDOWN_BE_BOUND_PERC"
};

static const char *top_down_l2[TOPDOWN_NUM_COUNTERS_LVL2] = {
	"TOPDOWN_HEAVY_OPS_PERC", "TOPDOWN_LIGHT_OPS_PERC",
	"TOPDOWN_BR_MISPREDICT_PERC", "TOPDOWN_MACHINE_CLEARS_PERC",
	"TOPDOWN_FETCH_LAT_PERC", "TOPDOWN_FETCH_BAND_PERC",
	"TOPDOWN_MEM_BOUND_PERC", "TOPDOWN_CORE_BOUND_PERC"
};

/* Identify the TopDown set by its first counter. */
static int HWCBE_PAPI_Is_TopDown_Set(char **counters)
{
	return topdown_level_global != TOPDOWN_DISABLED &&
	       counters != NULL && strcmp(counters[0], top_down_l1[0]) == 0;
}

/* Select the TopDown level and add its counters through the normal HWC set path. */
int HWCBE_PAPI_Add_TopDown_Set(unsigned level, int rank)
{
	char *topdown_counters[TOPDOWN_NUM_COUNTERS] = { NULL };
	unsigned i = 0;

	topdown_level_global =
		(level == TOPDOWN_LEVEL_1 || level == TOPDOWN_LEVEL_2) ? level : TOPDOWN_DISABLED;
	topdown_num_counters = (topdown_level_global == TOPDOWN_LEVEL_2) ? TOPDOWN_NUM_COUNTERS :
		(topdown_level_global == TOPDOWN_LEVEL_1) ? TOPDOWN_NUM_COUNTERS_LVL1 : 0;

	if (topdown_level_global == TOPDOWN_DISABLED)
		return 0;

	for (i = 0; i < TOPDOWN_NUM_COUNTERS_LVL1; i++)
		topdown_counters[i] = (char *) top_down_l1[i];

	if (topdown_level_global == TOPDOWN_LEVEL_2)
	{
		for (i = 0; i < TOPDOWN_NUM_COUNTERS_LVL2; i++)
			topdown_counters[TOPDOWN_NUM_COUNTERS_LVL1 + i] = (char *) top_down_l2[i];
	}

	return HWCBE_PAPI_Add_Set(HWC_num_sets + 1, rank, topdown_num_counters,
		topdown_counters, NULL, NULL, NULL, 0, NULL, NULL);
}

static void HWCBE_PAPI_AddDefinition (unsigned event_code, char *code, char *description)
{
	int found = FALSE;
	unsigned u;

	for (u = 0; !found && (u < num_hwc_used); u++)
		found = hwc_used[u].event_code == event_code;

	if (!found)
	{
		hwc_used = (HWC_Definition_t*) xrealloc (hwc_used,
			sizeof(HWC_Definition_t)*(num_hwc_used+1));
		hwc_used[num_hwc_used].event_code = event_code;
		snprintf (hwc_used[num_hwc_used].description,
			MAX_HWC_DESCRIPTION_LENGTH, "%s [%s]", code, description);
		num_hwc_used++;
	}
}

HWC_Definition_t *HWCBE_PAPI_GetCounterDefinitions(unsigned *count)
{
	*count = num_hwc_used;
	return hwc_used;
}

#if !defined(PAPIv3)
# error "-DNEW_HWC_SYSTEM requires PAPI v3 support"
#endif

int HWCBE_PAPI_Allocate_eventsets_per_thread (int num_set, int old_thread_num, int new_thread_num)
{
	int i;

	HWC_sets[num_set].eventsets = (int *) xrealloc (HWC_sets[num_set].eventsets, sizeof(int)*new_thread_num);

	for (i = old_thread_num; i < new_thread_num; i++)
		HWC_sets[num_set].eventsets[i] = PAPI_NULL;

	return TRUE;
}

#if defined(PAPI_SAMPLING_SUPPORT)
int Add_Overflows_To_Set (int rank, int num_set, int pretended_set, 
	int num_overflows, char **counter_to_ovfs, unsigned long long *ovf_values)
{
	int cnt, i, found;
	char *strtoul_check;

	HWC_sets[num_set].OverflowCounter = (int*) xmalloc (sizeof(int) * num_overflows);
	HWC_sets[num_set].OverflowValue = (long long*) xmalloc (sizeof(long long) * num_overflows);
	HWC_sets[num_set].NumOverflows = num_overflows;
	for (cnt = 0; cnt < num_overflows; cnt++)
	{
		char *counter_last_position = &(counter_to_ovfs[cnt][strlen(counter_to_ovfs[cnt])]);
	
		/* Convert this counter into a code */
		HWC_sets[num_set].OverflowCounter[cnt] = strtoul (counter_to_ovfs[cnt], &strtoul_check, 16);
		if (strtoul_check != counter_last_position)
		{
			int EventCode;
			if (PAPI_event_name_to_code(counter_to_ovfs[cnt], &EventCode) != PAPI_OK)
			{
				if (rank == 0)
					fprintf (stderr, PACKAGE_NAME": Cannot parse HWC %s in set %d for sampling, skipping\n", counter_to_ovfs[cnt], pretended_set);
				HWC_sets[num_set].OverflowCounter[cnt] = NO_COUNTER;
			}
			else
				HWC_sets[num_set].OverflowCounter[cnt] = EventCode;
		}

		/* Check if this counter code is in the HWC_set */
		if (HWC_sets[num_set].OverflowCounter[cnt] != NO_COUNTER)
		{
			for (found = FALSE, i = 0; i < HWC_sets[num_set].num_counters; i++)
				found |= (HWC_sets[num_set].counters[i] == HWC_sets[num_set].OverflowCounter[cnt]);

			if (!found)
			{
				HWC_sets[num_set].OverflowCounter[cnt] = NO_COUNTER;
				if (rank == 0)
					fprintf (stderr, PACKAGE_NAME": Sampling counter %s is not in available in set\n", counter_to_ovfs[cnt]);
				/* return FALSE; */
			}
		}

		HWC_sets[num_set].OverflowValue[cnt] = ovf_values[cnt];
	
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": HWC set %d sampling counter %s (0x%08x) every %lld events.\n", pretended_set, counter_to_ovfs[cnt], HWC_sets[num_set].OverflowCounter[cnt], ovf_values[cnt]);
	}


	return TRUE;
}
#endif

int HWCBE_PAPI_Add_Set (int pretended_set, int rank, int ncounters, char **counters,
	char *domain, char *change_at_globalops, char *change_at_time, 
	int num_overflows, char **overflow_counters, unsigned long long *overflow_values)
{
	int i, rc, num_set = HWC_num_sets;
	int is_topdown_set = HWCBE_PAPI_Is_TopDown_Set(counters);
	/* Keep normal sets limited to MAX_HWC and allow the full TopDown set. */
	int max_counters = is_topdown_set ? MAX_HWC_IN_SET : MAX_HWC;
	PAPI_event_info_t info;
#if !defined(PAPI_SAMPLING_SUPPORT)
	UNREFERENCED_PARAMETER(num_overflows);
	UNREFERENCED_PARAMETER(overflow_counters);
	UNREFERENCED_PARAMETER(overflow_values);
#endif
	
	if (ncounters == 0 || counters == NULL)
		return 0;
	
	if (ncounters > max_counters)
	{
		fprintf (stderr, PACKAGE_NAME": You cannot provide more HWC counters than %d (see set %d)\n", max_counters, pretended_set);
		ncounters = max_counters;
	}
	
	HWC_sets = (struct HWC_Set_t *) xrealloc (HWC_sets, sizeof(struct HWC_Set_t)* (HWC_num_sets+1));

	/* Initialize this set */
	HWC_sets[num_set].num_counters = 0;
	HWC_sets[num_set].eventsets = NULL;
#if defined(PAPI_SAMPLING_SUPPORT)
	HWC_sets[num_set].OverflowCounter = NULL;
	HWC_sets[num_set].OverflowValue = NULL;
	HWC_sets[num_set].NumOverflows = 0;
#endif

	/* Initialize every counter position as unused before adding the set. */
	for (i = 0; i < MAX_HWC_IN_SET; i++)
	{
		HWC_sets[num_set].counters[i] = NO_COUNTER;
	}

	for (i = 0; i < ncounters; i++)
	{
		/* counter_last_position will hold the address of the end of the 
		   counter[i] string 
		   This shall be compared with strtoul_check to know if the hex
		   is correct or not
		*/
		char *counter_last_position = &(counters[i][strlen(counters[i])]);
		char *strtoul_check;
		int counter_is_name;

		HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] = 
			strtoul (counters[i], &strtoul_check, 16);
		counter_is_name = (strtoul_check != counter_last_position);

		if (counter_is_name)
		{
			int EventCode;
			if (PAPI_event_name_to_code(counters[i], &EventCode) != PAPI_OK)
			{
				if (rank == 0)
					fprintf (stderr, PACKAGE_NAME": Cannot parse HWC %s in set %d, skipping\n", counters[i], pretended_set);
			}
			else
			{
				HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] = EventCode;
			}
		}

		rc = PAPI_get_event_info (HWC_sets[num_set].counters[HWC_sets[num_set].num_counters], &info);
		if (rc != PAPI_OK)
		{
			if (rank == 0)
				fprintf (stderr, PACKAGE_NAME": Error! Cannot query information for hardware counter %s (0x%08x). Check set %d.\n", counters[i], HWC_sets[num_set].counters[HWC_sets[num_set].num_counters], pretended_set);

			HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] = NO_COUNTER;
		}
		/* Native events seem that could have info.count = 0! */
		else if (rc == PAPI_OK && info.count == 0 && (HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] & PAPI_NATIVE_MASK) == 0)
		{
			if (rank == 0)
				fprintf (stderr, PACKAGE_NAME": Error! Hardware counter %s (0x%08x) is not available. Check set %d.\n", counters[i], HWC_sets[num_set].counters[HWC_sets[num_set].num_counters], pretended_set);

			HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] = NO_COUNTER;
		}
		else
		{
			if (rank == 0)
			{
				char counter_name[PAPI_MAX_STR_LEN];
				const char *counter_label = info.symbol;

				if (PAPI_event_code_to_name (HWC_sets[num_set].counters[HWC_sets[num_set].num_counters],
					counter_name) == PAPI_OK)
					counter_label = counter_name;

				HWCBE_PAPI_AddDefinition (HWC_sets[num_set].counters[HWC_sets[num_set].num_counters],
					counter_label, (info.event_code & PAPI_PRESET_MASK)?info.short_descr:info.long_descr);
			}

			HWC_sets[num_set].num_counters++;
		}
	}

	if (HWC_sets[num_set].num_counters == 0)
	{
		if (rank == 0)
			fprintf (stderr, PACKAGE_NAME": Set %d of counters seems to be empty/invalid, skipping\n", pretended_set);
		return 0;
	}

	/* Require all TopDown metrics because the merger labels them by fixed position. */
	if (is_topdown_set && HWC_sets[num_set].num_counters != ncounters)
	{
		if (rank == 0)
			fprintf (stderr, PACKAGE_NAME": TopDown requires all %d counters; disabling TopDown\n", ncounters);
		HWC_sets[num_set].num_counters = 0;
		return 0;
	}

	/* Just check if the user wants us to change the counters in some manner */
	if (change_at_time != NULL)
	{
		HWC_sets[num_set].change_at = __Extrae_Utils_getTimeFromStr (change_at_time, 
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
				fprintf (stdout, PACKAGE_NAME": PAPI domain set to ALL for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].domain = PAPI_DOM_ALL;
		}	
		else if (!strcasecmp(domain, "kernel"))
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PAPI domain set to KERNEL for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].domain = PAPI_DOM_KERNEL;
		}	
		else if (!strcasecmp(domain, "user"))
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PAPI domain set to USER for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].domain = PAPI_DOM_USER;
		}	
		else if (!strcasecmp(domain, "other"))
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PAPI domain set to OTHER for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].domain = PAPI_DOM_OTHER;
		}	
		else
		{
			if (rank == 0)
				fprintf (stdout, PACKAGE_NAME": PAPI domain set to USER for HWC set %d\n",
					pretended_set);
			HWC_sets[num_set].domain = PAPI_DOM_USER;
		}
	} /* domain != NULL */
	else
	{
		if (rank == 0)
			fprintf (stdout, PACKAGE_NAME": PAPI domain set to USER for HWC set %d\n",
				pretended_set);
		HWC_sets[num_set].domain = PAPI_DOM_USER;
	}

	HWCBE_PAPI_Allocate_eventsets_per_thread (num_set, 0, Backend_getNumberOfThreads());

	if (is_topdown_set)
	{
		/* Store the TopDown set index so it can remain active during rotation. */
		TopDown_set_index = num_set;
	}

	/* We validate this set */
	HWC_num_sets++;

	if (rank == 0)
	{
		fprintf (stdout, PACKAGE_NAME": HWC set %d contains following counters < ", pretended_set);
		for (i = 0; i < HWC_sets[num_set].num_counters; i++)
		{
			if (HWC_sets[num_set].counters[i] != NO_COUNTER)
			{
				char counter_name[PAPI_MAX_STR_LEN];

				PAPI_event_code_to_name (HWC_sets[num_set].counters[i], counter_name);
				fprintf (stdout, "%s (0x%08x) ", counter_name, HWC_sets[num_set].counters[i]);
			}
		}
		fprintf (stdout, ">");

		if (HWC_sets[num_set].change_type == CHANGE_TIME)
			fprintf (stdout, " - changing every %lld nanoseconds\n", HWC_sets[num_set].change_at);
		else if (HWC_sets[num_set].change_type == CHANGE_GLOPS)
			fprintf (stdout, " - changing every %lld global operations\n", HWC_sets[num_set].change_at);
		else
			fprintf (stdout, " - never changes\n");

		fflush (stdout);
	}

#if defined(PAPI_SAMPLING_SUPPORT)
	if (num_overflows > 0)
		Add_Overflows_To_Set (rank, num_set, pretended_set, num_overflows,
			overflow_counters, overflow_values);
#endif

	return HWC_sets[num_set].num_counters;
}

#if defined(PAPI_SAMPLING_SUPPORT)
void PAPI_sampling_handler (int EventSet, void *address, long_long overflow_vector, void *context)
{
	UNREFERENCED_PARAMETER(overflow_vector);
	UNREFERENCED_PARAMETER(context);
	UNREFERENCED_PARAMETER(EventSet);

	Extrae_SamplingHandler_PAPI(address);
}
#endif

int HWCBE_PAPI_Start_Set (UINT64 countglops, UINT64 time, int numset, int threadid)
{
#if defined(PAPI_SAMPLING_SUPPORT)
	int i;
#endif
	int rc;

	/* The given set is a valid one? */
	if (numset < 0 || numset >= HWC_num_sets)
		return FALSE;

	HWC_current_changeat = HWC_sets[numset].change_at;
	HWC_current_changetype = HWC_sets[numset].change_type;
	HWC_current_timebegin[threadid] = time;
	HWC_current_glopsbegin[threadid] = countglops;

	/* Mark this counter set as the current set */
	HWCEVTSET(threadid) = HWC_sets[numset].eventsets[threadid];

#if defined(PAPI_SAMPLING_SUPPORT)
	for (i = 0; i < HWC_sets[numset].NumOverflows; i++)
	{
		if (HWC_sets[numset].OverflowCounter[i] != NO_COUNTER)
		{
			rc = PAPI_overflow (HWCEVTSET(threadid), HWC_sets[numset].OverflowCounter[i],
			  HWC_sets[numset].OverflowValue[i], 0, PAPI_sampling_handler);
			if (rc < 0)
			{
				Extrae_setSamplingEnabled (FALSE);
				fprintf (stderr, PACKAGE_NAME": PAPI_overflow failed for thread %d - counter %x!\n", threadid, HWC_sets[numset].OverflowCounter[i]);
			}
			else
				Extrae_setSamplingEnabled (TRUE);
		}
	}
#endif

	rc = PAPI_start (HWCEVTSET(threadid));
 	if (rc == PAPI_OK)
	{
		TRACE_EVENT (time, HWC_CHANGE_EV, numset);

#if defined(PAPI_SAMPLING_SUPPORT)
		if (HWC_sets[numset].NumOverflows > 0)
		{
			long long overflow_values[MAX_HWC];

			HARDWARE_COUNTERS_OVERFLOW(HWC_sets[numset].num_counters, 
			                           HWC_sets[numset].counters, 
			                           HWC_sets[numset].NumOverflows, 
			                           HWC_sets[numset].OverflowCounter,
			                           overflow_values);

			TRACE_EVENT_AND_GIVEN_COUNTERS (time, HWC_SET_OVERFLOW_EV, 0, MAX_HWC, overflow_values);
		}
#endif
	}
	else
	{
		fprintf (stderr, PACKAGE_NAME": PAPI_start failed to start eventset %d on thread %d! (error = %d)\n", numset+1, threadid, rc);
		if (rc == PAPI_ESYS)
		{
			perror ("PAPI_start");
			fprintf (stderr, PACKAGE_NAME": errno = %d\n", errno);
		}
	}

	return rc == PAPI_OK;
}

int HWCBE_PAPI_Stop_Set (UINT64 time, int numset, int threadid)
{
	long long values[MAX_HWC_IN_SET] = { 0 };
	int rc;

	UNREFERENCED_PARAMETER(time);

	if (numset < 0 || numset >= HWC_num_sets)
		return FALSE;

	rc = PAPI_stop (HWC_sets[numset].eventsets[threadid], values);
	if (rc != PAPI_OK)
	{
		fprintf (stderr, PACKAGE_NAME": PAPI_stop failed for thread %d! (error = %d)\n", threadid, rc);
	}
	else if (rc == PAPI_ESYS)
	{
		perror ("PAPI_stop");
		fprintf (stderr, PACKAGE_NAME": errno = %d\n", errno);
	}

	return rc == PAPI_OK;
}

void HWCBE_PAPI_CleanUp (unsigned nthreads)
{
	if (PAPI_is_initialized())
	{
		int state;
		int i;
		unsigned t;

		for (i = 0; i < HWC_num_sets; i++)
		{
			for (t = 0; t < nthreads; t++)
			{
				/* Some threads may not have created this EventSet, so skip invalid entries. */
				if (HWC_sets[i].eventsets[t] != PAPI_NULL)
				{
					/* Stop each running EventSet, including TopDown, before destroying it. */
					if (PAPI_state (HWC_sets[i].eventsets[t], &state) == PAPI_OK &&
						(state & PAPI_RUNNING))
						PAPI_stop (HWC_sets[i].eventsets[t], NULL);

					PAPI_cleanup_eventset (HWC_sets[i].eventsets[t]);
					PAPI_destroy_eventset (&HWC_sets[i].eventsets[t]);
				}
			}
			xfree (HWC_sets[i].eventsets);
		}

#if defined(PAPI_SAMPLING_SUPPORT)
		for (i = 0; i < HWC_num_sets; i++)
		{
			/* Free extra allocated memory */
			if (HWC_sets[i].NumOverflows > 0)
			{
				xfree (HWC_sets[i].OverflowValue);
				xfree (HWC_sets[i].OverflowCounter);
			}
		}
#endif
		xfree (HWC_sets); 

		PAPI_shutdown();
	}
}

/******************************************************************************
 **      Function name : PAPI_Initialize
 **
 **      Description :
 ******************************************************************************/

void HWCBE_PAPI_Initialize (int TRCOptions)
{
	UNREFERENCED_PARAMETER(TRCOptions);

	int rc;
	void *thread_identifier_function;

	/* PAPI initialization */
	rc = PAPI_library_init (PAPI_VER_CURRENT);
	if (rc != PAPI_VER_CURRENT)
	{
		if (rc > 0)
		{
			fprintf (stderr,
				PACKAGE_NAME": PAPI library version mismatch!\n"
				"          "PACKAGE_NAME" is compiled against PAPI v%d.%d , and \n"
				"          PAPI_library_init reported v%d.%d ,\n"
				"          Check that LD_LIBRARY_PATH points to the correct PAPI library.\n",
				PAPI_VERSION_MAJOR(PAPI_VER_CURRENT),
				PAPI_VERSION_MINOR(PAPI_VER_CURRENT),
				PAPI_VERSION_MAJOR(rc),
				PAPI_VERSION_MINOR(rc));
		}
		fprintf (stderr, PACKAGE_NAME": Can't use hardware counters!\n");
		fprintf (stderr, PACKAGE_NAME": PAPI library error: %s\n", PAPI_strerror (rc));

		if (rc == PAPI_ESYS)
			perror (PACKAGE_NAME": PAPI system error is ");

		return;
	}

#if defined(PAPI_SAMPLING_SUPPORT)
	/* Use any kind of sampling -- software or hardware */
	SamplingSupport = TRUE;
#endif

	thread_identifier_function = Extrae_get_thread_number_function();

	if (thread_identifier_function != NULL)
	{
		if ((rc = PAPI_thread_init ((unsigned long (*)(void)) thread_identifier_function)) != PAPI_OK)
		{
			fprintf (stderr, PACKAGE_NAME": PAPI_thread_init failed! Reason: %s\n", PAPI_strerror(rc));
			return;
		}
	}
}

int HWCBE_PAPI_Init_Thread (UINT64 time, int threadid, int forked)
{
	int i = 0;
	int j = 0;
	int rc = PAPI_OK;

	if (HWC_num_sets <= 0)
		return FALSE;

	/* A forked child must recreate the PAPI EventSets inherited from its parent. */
	if (forked)
	{
		for (i = 0; i < HWC_num_sets; i++)
		{
			/* Recreate each inherited EventSet after fork because PAPI cannot reuse it. */
			if (HWC_sets[i].eventsets[threadid] != PAPI_NULL)
			{
				PAPI_stop (HWC_sets[i].eventsets[threadid], NULL);
				rc = PAPI_cleanup_eventset (HWC_sets[i].eventsets[threadid]);
				if (rc == PAPI_OK)
					PAPI_destroy_eventset (&HWC_sets[i].eventsets[threadid]);
			}

			HWC_sets[i].eventsets[threadid] = PAPI_NULL;
		}
	}

	//if (!forked)
	{
		for (i = 0; i < HWC_num_sets; i++)
		{
			/* Create the eventset. Each thread will create its own eventset */
			rc = PAPI_create_eventset (&(HWC_sets[i].eventsets[threadid]));
			if (PAPI_OK != rc)
			{
				fprintf (stderr, PACKAGE_NAME": Error! Unable to create eventset (%d of %d) in task %d, thread %d\n", i+1, HWC_num_sets, TASKID, threadid);
				continue;
			}

			/* Add the selected counters */
			for (j = 0; j < HWC_sets[i].num_counters; j++)
			{
				if (HWC_sets[i].counters[j] != NO_COUNTER)
				{
					rc = PAPI_add_event (HWC_sets[i].eventsets[threadid], HWC_sets[i].counters[j]);
					if (rc != PAPI_OK)
					{
						char EventName[PAPI_MAX_STR_LEN];

						PAPI_event_code_to_name (HWC_sets[i].counters[j], EventName);
						fprintf (stderr, PACKAGE_NAME": Error! Hardware counter %s (0x%08x) cannot be added in set %d (task %d, thread %d)\n", EventName, HWC_sets[i].counters[j], i+1, TASKID, threadid);
						fprintf (stderr, "PAPI error %d: %s\n", rc, PAPI_strerror(rc));
						if (i == TopDown_set_index)
						{
							/* Disable only this thread's TopDown EventSet without changing the shared counter layout. */
							PAPI_cleanup_eventset (HWC_sets[i].eventsets[threadid]);
							PAPI_destroy_eventset (&HWC_sets[i].eventsets[threadid]);
							break;
						}
						HWC_sets[i].counters[j] = NO_COUNTER;

						/* If a counter fails to enter the EventSet, we can't just mark it
						 * as NO_COUNTER, because the next counter that enters the EventSet
						 * will take this one's position, and then we'll have a shift
						 * between the counters listed in HWC_sets, and the counters in the
						 * EventSet and the values read at PAPI_read. 
						 * Quick hack: Send the failing counter to the end of the
						 * HWC_sets.counters array and shift the remaining to the left.
						  
						                     __ Shift starting from here till the end of the array to the left, replacing the counter that failed
						                    |             __ As we shift to the left, we leave the last slot open, we need to mark it as NO_COUNTER
						                    |            |
						                    v            v
						[ HWC1 | HWC2 |  X  | HWC4 | ... | NO_COUNTER ] 
						                 ^
						                 |_ First counter that fails to enter the EventSet 
						 */

						/* Remove the failed counter by shifting later entries left within the set capacity. */
						int k = 0;
						int max_cnt = (i == TopDown_set_index) ? MAX_HWC_IN_SET : MAX_HWC;
						for (k = j; k < max_cnt - 1; k++) {
							
							HWC_sets[i].counters[k] = HWC_sets[i].counters[k+1];
						}
						HWC_sets[i].counters[max_cnt-1] = NO_COUNTER;
						HWC_sets[i].num_counters --;
						j --;
					}
				}
			}

			/* Skip this EventSet if it was not created. */
			if (HWC_sets[i].eventsets[threadid] == PAPI_NULL)
				continue;

			// This used to be in Backend_preInitialize for the main thread to emit the HWC_DEF_EV events,
			// right before the call to HWC_Start_Counters that ends up here. 
			// In the loop above we are still flagging NO_COUNTER's, thus the HWC_DEF_EV can't be emitted until now.
			// This function also calls HWCBE_PAPI_Start_Set, which in turn emits
			// other events that rely on having HWC_DEF_EV first, so this has to
			// happen somewhere in the middle. 
			// FIXME: The emission should happen from the common interface, not from
			// the PAPI backend (missing for PMAPI).
			/* Do not write internal TopDown counters as a normal HWC definition set. */
			if (threadid == 0 && i != TopDown_set_index)
			{
			  /* Write hardware counters set definitions (i.e. those that were
			   * succesfully added into PAPI EventSets) into the .mpit files*/
				int *HWCid;
				HWC_Get_Set_Counters_Ids (i, &HWCid); /* HWCid is allocated up to MAX_HWC and sets NO_COUNTER where appropriate */
				TRACE_EVENT_AND_GIVEN_COUNTERS (LAST_READ_TIME, HWC_DEF_EV, i, MAX_HWC, HWCid);
				xfree (HWCid);
			}

			/* PAPI domains only apply to the CPU perf_event component. */
#if defined(PAPI_VERSION) && (PAPI_VERSION_MAJOR(PAPI_VERSION) >= 4)
			{
				int cid = PAPI_get_eventset_component(HWC_sets[i].eventsets[threadid]);
				const PAPI_component_info_t *info = PAPI_get_component_info(cid);

				if (strcmp(info->name, "perf_event") != 0)
					continue;
			}
#endif
			/* Only CPU perf_event sets reach this point. Keep options local
			 * because it is only needed for the domain setup. */
			{
				PAPI_option_t options;

				xmemset (&options, 0, sizeof(options));
				options.domain.eventset = HWC_sets[i].eventsets[threadid];
				options.domain.domain = HWC_sets[i].domain;
				rc = PAPI_set_opt (PAPI_DOMAIN, &options);
				if (rc != PAPI_OK)
					fprintf (stderr, PACKAGE_NAME": Error when setting domain for eventset %d\n", i+1);
			}
		}
	} /* forked */ 

	if (HWC_TopDown_Enabled())
	{
		/* Start the permanent TopDown EventSet for this thread. */
		int topdown_eventset = HWC_sets[TopDown_set_index].eventsets[threadid];

		if (topdown_eventset != PAPI_NULL)
		{
			rc = PAPI_start (topdown_eventset);
			if (rc != PAPI_OK)
			{
				/* Discard the EventSet if it cannot be started. */
				fprintf (stderr, PACKAGE_NAME": Error starting TopDown PAPI EventSet for task %d, thread %d (%s)\n", TASKID, threadid, PAPI_strerror(rc));
				PAPI_cleanup_eventset (topdown_eventset);
				PAPI_destroy_eventset (&HWC_sets[TopDown_set_index].eventsets[threadid]);
			}
		}
	}
	/* Start the rotating set only when normal counters are configured. */
	if (HWC_num_rotating_sets > 0)
		HWC_Thread_Initialized[threadid] = HWCBE_PAPI_Start_Set (0, time, HWC_current_set[threadid], threadid);
	else
		HWC_Thread_Initialized[threadid] = HWC_TopDown_Enabled() &&
			(HWC_sets[TopDown_set_index].eventsets[threadid] != PAPI_NULL);

#if defined(ENABLE_PEBS_SAMPLING)                                               
	    Extrae_IntelPEBS_startSampling();                                              
#endif                                                                          

	return HWC_Thread_Initialized[threadid];
}

/* Read TopDown counters and emit their packed events. */
int HWCBE_PAPI_Emit_TopDown_Counters (unsigned int tid, UINT64 time)
{
	int topdown_eventset = PAPI_NULL;
	long long raw_values[MAX_HWC_IN_SET] = { 0 };
	long long scaled_values[MAX_HWC_IN_SET];
	int counter = 0;
	int num_counters = 0;
	int topdown_level = TOPDOWN_DISABLED;

	if (!HWC_TopDown_Enabled())
		return TRUE;

	/* Zero is a valid percentage; NO_COUNTER marks unused slots for the merger. */
	for (counter = 0; counter < MAX_HWC_IN_SET; counter++)
		scaled_values[counter] = NO_COUNTER;

	topdown_eventset = HWC_sets[TopDown_set_index].eventsets[tid];
	if (topdown_eventset == PAPI_NULL)
		return FALSE;

	if (PAPI_read(topdown_eventset, raw_values) != PAPI_OK)
	{
		fprintf (stderr, PACKAGE_NAME": TopDown PAPI_read failed for thread %d evtset %d\n",
			tid, topdown_eventset);
		return FALSE;
	}
	num_counters = HWC_sets[TopDown_set_index].num_counters;
	for (counter = 0; counter < num_counters; counter++)
	{
		double val = 0.0;

		/* WARNING: PAPI stores TopDown percentages as double bits in long long slots; a cast is incorrect. */
		memcpy(&val, &raw_values[counter], sizeof(val));
		if (val < 0.0)
			val = 0.0;
		else if (val > 100.0)
			val = 100.0;
		scaled_values[counter] = (long long)(val * 1000.0);
	}

	topdown_level = (num_counters == TOPDOWN_NUM_COUNTERS) ? TOPDOWN_LEVEL_2 : TOPDOWN_LEVEL_1;
	/* Always emit the 4 Level-1 metrics as one packed internal event. */
	TRACE_EVENT_AND_GIVEN_COUNTERS(time, TOPDOWN_PACKED_L1_EV, topdown_level,
		TOPDOWN_NUM_COUNTERS_LVL1, scaled_values);

	if (topdown_level == TOPDOWN_LEVEL_2)
	{
		/* Emit the 8 Level-2 metrics as a second packed internal event. */
		TRACE_EVENT_AND_GIVEN_COUNTERS(time, TOPDOWN_PACKED_L2_EV, topdown_level,
			TOPDOWN_NUM_COUNTERS_LVL2, (scaled_values + TOPDOWN_NUM_COUNTERS_LVL1));
	}

	return TRUE;
}

#if defined(IS_BG_MACHINE)
int __in_PAPI_read_BG = FALSE;
#endif
/* Read the current rotating counter set. */
int HWCBE_PAPI_Read (unsigned int tid, long long *store_buffer)
{
#if !defined(IS_BG_MACHINE)
	int rotating_counters_valid = TRUE;
	int i = 0;

	/* TopDown-only configurations have no normal rotating EventSet to read. */
	if (HWC_num_rotating_sets > 0)
	{
		int rotating_eventset = HWCEVTSET(tid);

		if (PAPI_read(rotating_eventset, store_buffer) != PAPI_OK)
		{
			fprintf (stderr, PACKAGE_NAME": PAPI_read failed for thread %d evtset %d (%s:%d)\n",
				tid, rotating_eventset, __FILE__, __LINE__);
			for (i = 0; i < MAX_HWC; i++)
				store_buffer[i] = NO_COUNTER;
			rotating_counters_valid = FALSE;
		}
	}
	else
	{
		/* Keep the normal HWC buffer explicit in TopDown-only mode. */
		for (i = 0; i < MAX_HWC; i++)
			store_buffer[i] = NO_COUNTER;
	}

	return rotating_counters_valid;
#else
	int rotating_eventset = PAPI_NULL;

	if (HWC_num_rotating_sets > 0)
		rotating_eventset = HWCEVTSET(tid);

	if (rotating_eventset != PAPI_NULL && !__in_PAPI_read_BG)
	{
		__in_PAPI_read_BG = TRUE;
		if (PAPI_read(rotating_eventset, store_buffer) != PAPI_OK)
		{
			fprintf (stderr, PACKAGE_NAME": PAPI_read failed for thread %d evtset %d (%s:%d)\n",
				tid, rotating_eventset, __FILE__, __LINE__);
			return 0;
		}
		__in_PAPI_read_BG = FALSE;
		return 1;
	}
	else
		return 0;
#endif
}

int HWCBE_PAPI_Reset (unsigned int tid)
{
	int reset_succeeded = TRUE;

	/* Reset the normal rotating EventSet only when it exists. */
	if (HWC_num_rotating_sets > 0 && PAPI_reset(HWCEVTSET(tid)) != PAPI_OK)
	{
		fprintf (stderr, PACKAGE_NAME": PAPI_reset failed for thread %d evtset %d (%s:%d)\n", \
			tid, HWCEVTSET(tid), __FILE__, __LINE__);
		reset_succeeded = FALSE;
	}
	return reset_succeeded;
}

/* Accumulation is only valid for the normal rotating EventSet. */
int HWCBE_PAPI_Accum (unsigned int tid, long long *store_buffer)
{
	if (HWC_num_rotating_sets <= 0)
		return FALSE;

	if (PAPI_accum(HWCEVTSET(tid), store_buffer) != PAPI_OK)
	{
		fprintf (stderr, PACKAGE_NAME": PAPI_accum failed for thread %d evtset %d (%s:%d)\n", \
			tid, HWCEVTSET(tid), __FILE__, __LINE__);
		return 0;		
	}
	return 1;
}
