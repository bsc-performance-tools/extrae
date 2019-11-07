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
#include <sys/time.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_STRINGS_H
#include <strings.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "utils.h"
#include "events.h"
#include "clock.h"
#include "threadid.h"
#include "record.h"
#include "trace_macros.h"
#include "wrapper.h"
#include "stdio.h"
#include "common_hwc.h"
#include "l4stat_hwc.h"
#include <bsp.h> /* for device driver prototypes */
#include <bsp/l4stat.h>

int sw_threads[8];


/*------------------------------------------------ Static Variables ---------*/

static HWC_Definition_t *hwc_used = NULL;
static unsigned num_hwc_used = 0;

static void HWCBE_L4STAT_AddDefinition(unsigned event_code, char *description)
{
	int found = FALSE;
	unsigned u;

	for (u = 0; !found && (u < num_hwc_used); u++)
		found = hwc_used[u].event_code == event_code;

	if (!found)
	{
		hwc_used = (HWC_Definition_t *)realloc(hwc_used,
											   sizeof(HWC_Definition_t) * (num_hwc_used + 1));
		if (hwc_used == NULL)
		{
			fprintf(stderr, "ERROR! Cannot allocate memory to add definitions for hardware counters\n");
			return;
		}
		hwc_used[num_hwc_used].event_code = event_code;
		snprintf(hwc_used[num_hwc_used].description,
				 MAX_HWC_DESCRIPTION_LENGTH, "[%s]", description);
		num_hwc_used++;
	}
}

HWC_Definition_t *HWCBE_L4STAT_GetCounterDefinitions(unsigned *count)
{
	*count = num_hwc_used;
	return hwc_used;
}

int HWCBE_L4STAT_Add_Set(int pretended_set, int rank, int ncounters, char **counters,
						 char *domain, char *change_at_globalops, char *change_at_time,
						 int num_overflows, char **overflow_counters, unsigned long long *overflow_values)
{
	int i, rc, num_set = HWC_num_sets;
	char *info;

	UNREFERENCED_PARAMETER(num_overflows);
	UNREFERENCED_PARAMETER(overflow_counters);
	UNREFERENCED_PARAMETER(overflow_values);

	if (ncounters == 0 || counters == NULL)
		return 0;

	if (ncounters > MAX_HWC)
	{
		fprintf(stderr, PACKAGE_NAME ": You cannot provide more HWC counters than %d (see set %d)\n", MAX_HWC, pretended_set);
		ncounters = MAX_HWC;
	}

	HWC_sets = (struct HWC_Set_t *)realloc(HWC_sets, sizeof(struct HWC_Set_t) * (HWC_num_sets + 1));
	if (HWC_sets == NULL)
	{
		fprintf(stderr, PACKAGE_NAME ": Cannot allocate memory for HWC_set (rank %d)\n", rank);
		return 0;
	}

	/* Initialize this set */
	HWC_sets[num_set].num_counters = 0;

	for (i = 0; i < ncounters; i++)
	{
		/* counter_last_position will hold the address of the end of the
		   counter[i] string
		   This shall be compared with strtoul_check to know if the hex
		   is correct or not
		*/
		char *counter_last_position = &(counters[i][strlen(counters[i])]);
		char *strtoul_check;

		HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] =
			strtoul(counters[i], &strtoul_check, 16);

		if (strtoul_check != counter_last_position)
		{
			int EventCode;

			if (rank == 0)
				fprintf(stderr, PACKAGE_NAME ": Currently name translation of counters is disabled, please specify the counter number \n");
			continue;
		}
		else if (HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] > 0x9F || HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] < 0x0)
		{
			if (rank == 0)
				fprintf(stderr, PACKAGE_NAME ": Wrong counter number %s, please specify a counter in the range 0x0 to 0x9F\n", counters[i]);
			continue;
		}

		info = l4stat_event_names[HWC_sets[num_set].counters[HWC_sets[num_set].num_counters]];
		if (strcmp(info, L4STAT_BAD_CMD) == 0)
		{
			if (rank == 0)
				fprintf(stderr, PACKAGE_NAME ": Error! Cannot query information for hardware counter %s (0x%08x). Check set %d.\n", counters[i], HWC_sets[num_set].counters[HWC_sets[num_set].num_counters], pretended_set);

			HWC_sets[num_set].counters[HWC_sets[num_set].num_counters] = NO_COUNTER;
		}
		else
		{
			if (rank == 0)
				HWCBE_L4STAT_AddDefinition(HWC_sets[num_set].counters[HWC_sets[num_set].num_counters], info);

			HWC_sets[num_set].num_counters++;
		}
	}

	if (HWC_sets[num_set].num_counters == 0)
	{
		if (rank == 0)
			fprintf(stderr, PACKAGE_NAME ": Set %d of counters seems to be empty/invalid, skipping\n", pretended_set);
		return 0;
	}

	HWC_sets[num_set].change_type = CHANGE_NEVER;

	//HWCBE_PAPI_Allocate_eventsets_per_thread (num_set, 0, Backend_getNumberOfThreads());

	/* We validate this set */
	HWC_num_sets++;

	if (rank == 0)
	{
		fprintf(stdout, PACKAGE_NAME ": HWC set %d contains following counters < ", pretended_set);
		for (i = 0; i < HWC_sets[num_set].num_counters; i++)
		{
			if (HWC_sets[num_set].counters[i] != NO_COUNTER)
			{
				fprintf(stdout, "%s (0x%08x) ", l4stat_event_names[HWC_sets[num_set].counters[i]], HWC_sets[num_set].counters[i]);
			}
		}
		fprintf(stdout, ">");

		if (HWC_sets[num_set].change_type == CHANGE_TIME)
			fprintf(stdout, " - changing every %lld nanoseconds\n", HWC_sets[num_set].change_at);
		else if (HWC_sets[num_set].change_type == CHANGE_GLOPS)
			fprintf(stdout, " - changing every %lld global operations\n", HWC_sets[num_set].change_at);
		else
			fprintf(stdout, " - never changes\n");

		fflush(stdout);
	}

	return HWC_sets[num_set].num_counters;
}

int HWCBE_L4STAT_Start_Set(UINT64 countglops, UINT64 time, int numset, int threadid)
{

	int rc;

	/* The given set is a valid one? */
	if (numset < 0 || numset >= HWC_num_sets)
		return FALSE;

	HWC_current_changeat = HWC_sets[numset].change_at;
	HWC_current_changetype = HWC_sets[numset].change_type;
	HWC_current_timebegin[threadid] = time;
	HWC_current_glopsbegin[threadid] = countglops;

	TRACE_EVENT(time, HWC_CHANGE_EV, numset);

	return TRUE;
}

int HWCBE_L4STAT_Stop_Set(UINT64 time, int numset, int threadid)
{
	//Not required currently
	return 1;
}

void HWCBE_L4STAT_CleanUp(unsigned nthreads)
{
	int ret;
	int i;

	for (i = 0; i < 16; i++)
	{
		ret = l4stat_counter_clear(i);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat clear!\n");
			exit(1);
		}
		ret = l4stat_counter_disable(i);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat disable!\n");
			exit(1);
		}
	}
}

/******************************************************************************
 **      Function name : L4STAT_Initialize
 **
 **      Description :
 ******************************************************************************/

void HWCBE_L4STAT_Initialize(int TRCOptions)
{
	int ret;
	int i, j, counter;
	for (i = 0; i < 16; i++)
	{
		ret = l4stat_counter_clear(i);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat clear!\n");
			exit(1);
		}
		ret = l4stat_counter_disable(i);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat disable!\n");
			exit(1);
		}
	}

	for (i = 0; i < 4; i++)
	{

		counter = HWC_sets[0].num_counters * i;

		for (j = 0; j < HWC_sets[0].num_counters; j++)
		{

			//printf("Enabling counter %i, cpu %i event: %i \n", counter, i,  HWC_sets[0].counters[j] );
			ret = l4stat_counter_enable(counter, HWC_sets[0].counters[j], i, 0);
			if (ret != L4STAT_ERR_OK)
			{
				printf("Error: l4stat enable!\n");
				exit(1);
			}
			ret = l4stat_counter_set(counter, 0);
			if (ret != L4STAT_ERR_OK)
			{
				printf("Error: l4stat init thread!\n");
				return -1;
			}
			counter++;
		}
	}
}

int HWCBE_L4STAT_Init_Thread(UINT64 time, int threadid, int forked)
{
	int cpu_self, i, ret;
	cpu_self = (int)rtems_get_current_processor();
	sw_threads[threadid] = cpu_self;
	HWC_Thread_Initialized[threadid] = HWCBE_L4STAT_Start_Set(0, time, HWC_current_set[threadid], threadid);

	return HWC_Thread_Initialized[threadid];
}

int HWCBE_L4STAT_Read(unsigned int tid, long long *store_buffer)
{
	int cpu_self, i, ret;
	uint32_t value;
	cpu_self = (int)rtems_get_current_processor();
	int counter = HWC_sets[0].num_counters * cpu_self;
	for (i = 0; i < HWC_sets[0].num_counters; i++)
	{
		ret = l4stat_counter_get(counter, &value);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat read thread!\n");
			return -1;
		}
		counter++;
		store_buffer[i] = (long long)value;
	}
	return TRUE;
}

int HWCBE_L4STAT_Read_Sampling(unsigned int tid, long long *store_buffer)
{
	int cpu_self, i, ret;
	uint32_t value;
	cpu_self = sw_threads[tid];
	int counter = HWC_sets[0].num_counters * cpu_self;
	for (i = 0; i < HWC_sets[0].num_counters; i++)
	{
		ret = l4stat_counter_get(counter, &value);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat read thread!\n");
			return -1;
		}

		store_buffer[i] = value;
		ret = l4stat_counter_set(counter, 0);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat reset thread!\n");
			return -1;
		}
		counter++;
	}

	return TRUE;
}

int HWCBE_L4STAT_Reset(unsigned int tid)
{
	int cpu_self, i, ret;
	uint32_t value;
	cpu_self = (int)rtems_get_current_processor();
	int counter = HWC_sets[0].num_counters * cpu_self;
	for (i = 0; i < HWC_sets[0].num_counters; i++)
	{
		ret = l4stat_counter_set(counter, 0);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat reset thread!\n");
			return -1;
		}
		counter++;
	}
	return TRUE;
}

void HWCBE_L4STAT_Update_Sampling_Cores(unsigned int tid)
{
	sw_threads[tid] = (int)rtems_get_current_processor();
}

int HWCBE_L4STAT_Accum(unsigned int tid, long long *store_buffer)
{
	int cpu_self, i, ret;
	uint32_t value;
	cpu_self = (int)rtems_get_current_processor();
	int counter = HWC_sets[0].num_counters * cpu_self;
	for (i = 0; i < HWC_sets[0].num_counters; i++)
	{
		ret = l4stat_counter_get(counter, &value);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat read accum thread!\n");
			return -1;
		}
		ret = l4stat_counter_set(counter, 0);
		if (ret != L4STAT_ERR_OK)
		{
			printf("Error: l4stat reset accum thread!\n");
			return -1;
		}
		counter++;
		store_buffer[i] += value;
	}
	return TRUE;
}
