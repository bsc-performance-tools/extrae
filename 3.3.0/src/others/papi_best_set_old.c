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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#include <papi.h>

int* CreateBitmask (int numberofbits)
{
	int i;
	int *result;

	result = (int*) malloc (numberofbits*sizeof(int));
	if (result == NULL)
	{
		fprintf (stderr, "Error! Unable to allocate bitmask memory! Dying.\n");
		exit (-5);
	}

	for (i = 0; i < numberofbits; i++)
		result[i] = 0;

	return result;
}

void Bitmask_Copy (int *source, int *destination, int numberofbits)
{
	int i;

	for (i = 0; i < numberofbits; i++)
		destination[i] = source[i];
}

void Bitmask_PrintValue (int *bitmask, int numberofbits)
{
	int i;

	for (i = 0; i < numberofbits; i++)
		printf ("%c", (bitmask[i]==0)?'0':'1');
	printf ("\n");
}

void Bitmask_NextValue (int *bitmask, int numberofbits)
{
	int i;

	i = 0;
	while ((bitmask[i] == 1) && (i < numberofbits))
		bitmask[i++] = 0;
	if (i < numberofbits)
		bitmask[i] = 1;
}

int Bitmask_CountActiveBits (int *bitmask, int numberofbits)
{
	int active;
	int i;

	for (active = 0, i = 0; i < numberofbits; i++)
		if (bitmask[i])
			active++;
	return active;
}

int Bitmask_isLastValue (int *bitmask, int numberofbits)
{	
	int i;
	int isLast = TRUE;

	for (i = 0; i < numberofbits; i++)
		isLast = isLast && bitmask[i];
	return isLast;
}

int CheckBitmapCounters (int ncounters, unsigned *counters, int *bitmask)
{
	int valid_bitmap = TRUE;
	int EventSet = PAPI_NULL;
	int i;

	if (PAPI_create_eventset(&EventSet) != PAPI_OK)
	{
		fprintf (stderr, "Error! Failed to create the eventset!\n");
		return FALSE;
	}

	for (i = 0; i < ncounters; i++)
		if (bitmask[i])
			valid_bitmap = valid_bitmap && (PAPI_add_event (EventSet, counters[i]) == PAPI_OK);

	if (PAPI_cleanup_eventset (EventSet) != PAPI_OK)
		fprintf (stderr, "Error! Failed to cleanup the eventset\n");

	if (PAPI_destroy_eventset (&EventSet) != PAPI_OK)
		fprintf (stderr, "Error! Failed to destroy the eventset\n");

	return valid_bitmap;
}

/*   CheckMaxEventSet

     Search the eventset with the maximum events added.
     Uses a bitmask to generate all the possible combinations.
*/
void CheckMaxEventSet (int ncounters, unsigned *counters)
{
	int i;
	int j;
	int k;
	int rc;
	int *bitmask;
	int **maxs;
	int nmaxs;
	char EventName[PAPI_MAX_STR_LEN];

	bitmask = CreateBitmask (ncounters);
	nmaxs = 0;
	maxs = NULL;

	do
	{
		Bitmask_NextValue (bitmask, ncounters);
		if (CheckBitmapCounters (ncounters, counters, bitmask))
		{
			if (nmaxs > 0)
			{
				/* This set has more counters than the rest of sets.
				   Remove them and add this one! */
				if (Bitmask_CountActiveBits (bitmask, ncounters) > Bitmask_CountActiveBits (maxs[0], ncounters))
				{
					int i;
					for (i = 0; i < nmaxs; i++)
						free (maxs[i]);

					maxs = (int **) realloc (maxs, (sizeof(int*)));
					if (maxs == NULL)
					{
						fprintf (stderr, "Error! Cannot re-allocate memory for maximum eventsets\n");
						exit (-6);
					}
					maxs[0] = CreateBitmask (ncounters);
					Bitmask_Copy (bitmask, maxs[0], ncounters);
					nmaxs = 1;
				}
				/* This set has the same number of counters than maximum set. Add it */
				else if (Bitmask_CountActiveBits (bitmask, ncounters) == Bitmask_CountActiveBits (maxs[0], ncounters))
				{
					maxs = (int **) realloc (maxs, ((nmaxs+1)*sizeof(int*)));
					if (maxs == NULL)
					{
						fprintf (stderr, "Error! Cannot re-allocate memory for maximum eventsets\n");
						exit (-6);
					}
					maxs[nmaxs] = CreateBitmask (ncounters);
					Bitmask_Copy (bitmask, maxs[nmaxs], ncounters);
					nmaxs++;
				}
			}
			else
			{
				maxs = (int **) malloc (sizeof(int*));
				if (maxs == NULL)
				{
					fprintf (stderr, "Error! Cannot allocate memory for maximum eventsets\n");
					exit (-6);
				}
				maxs[0] = CreateBitmask (ncounters);
				Bitmask_Copy (bitmask, maxs[0], ncounters);
				nmaxs = 1;
			}
		}
	}
	while (!Bitmask_isLastValue(bitmask, ncounters));

#if 0
	fprintf (stdout, "** Number of combinations found: %d\n", nmaxs);
	fprintf (stdout, "** Maximum events placed per combination: %d\n", Bitmask_CountActiveBits (maxs[0], ncounters));
#endif

	fprintf (stdout, "Combinations maximizing the number of compatible counters:\n");
	for (k = 0; k < nmaxs; k++)
	{
		fprintf (stdout, "* Combination #%d (set of %d): ", k+1, Bitmask_CountActiveBits (maxs[k], ncounters));
		for (j = 0, i = 0; i < ncounters; i++)
			if (maxs[k][i])
			{
				rc = PAPI_event_code_to_name (counters[i], EventName);
				if (rc == PAPI_OK)
					fprintf (stdout, "%s%s", j>0?",":"", EventName);
				else
					fprintf (stdout, "%s%08x", j>0?",":"", counters[i]);
				j++;
			}
		fprintf (stdout, "\n");
	}
}

/*   CheckInOrder

     Check if we can add all the counters in the order given.
     This will just iterate through all counters trying to add them in the
     eventset.
*/
void CheckInOrder (int ncounters, unsigned *counters)
{
	int rc;
	int i;
	int naddedcounters;
	int EventSet = PAPI_NULL;
	char EventName[PAPI_MAX_STR_LEN];

#if 0
	fprintf (stdout, "\n** Checking in order the following counters: ");
	for (i = 0; i < ncounters; i++)
	{
		rc = PAPI_event_code_to_name (counters[i], EventName);
		if (rc == PAPI_OK)
			fprintf (stdout, "%s%s", i>0?",":"", EventName);
		else
			fprintf (stdout, "%s%08x", i>0?",":"", counters[i]);
	}
	fprintf (stdout, "\n");
#endif

	if (PAPI_create_eventset(&EventSet) != PAPI_OK)
	{
		fprintf (stderr, "Error! Failed to create the eventset!\n");
		return;
	}

	fprintf (stdout, "Suggested compatible counters preserving the given priority:\n");
	for (naddedcounters = 0, i = 0; i < ncounters; i++)
	{
		if (PAPI_add_event (EventSet, counters[i]) == PAPI_OK)
		{
			rc = PAPI_event_code_to_name (counters[i], EventName);
			if (rc == PAPI_OK)
				fprintf (stdout, "%s%s", i>0?",":"", EventName);
			else
				fprintf (stdout, "%s%08x", i>0?",":"", counters[i]);
			naddedcounters++;
		}
	}

	if (naddedcounters > 0)
		fprintf (stdout, " (set of %d counters)\n\n", naddedcounters);
	else
		fprintf (stdout, "NONE!\n\n");

	if (PAPI_cleanup_eventset(EventSet) != PAPI_OK)
		fprintf (stderr, "Error! Failed to cleanup the eventset\n");
	if (PAPI_destroy_eventset (&EventSet) != PAPI_OK)
		fprintf (stderr, "Error! Failed to destroy the eventset\n");
}


int main (int argc, char *argv[])
{
	int num_events;
	unsigned* events;
	int i;
	int rc;

	if (argc < 2)
	{
		fprintf (stderr, "Error: You must provide a set of PAPI counters\n");
		return -1;
	}

	rc = PAPI_library_init(PAPI_VER_CURRENT);
	if (rc != PAPI_VER_CURRENT && rc > 0)
	{
		fprintf (stderr, "Error: PAPI library version mismatch!\n");
		return -2;
	}

	events = (unsigned*) malloc (sizeof(unsigned)*(argc-1));
	if (events == NULL)
	{
		fprintf (stderr, "Error: Cannot allocate memory for %d events\n", argc-1);
		return -3;
	}

	fprintf (stdout, "This binary was built using PAPI found in %s\n", PAPI_HOME);

#if 0
	fprintf (stdout, "\nNumber Hardware Counters : %d\n", PAPI_get_opt(PAPI_MAX_HWCTRS, NULL));
	fprintf (stdout, "Max Multiplex Counters   : %d\n", PAPI_get_opt(PAPI_MAX_MPX_CTRS, NULL));
#endif
	fprintf (stdout, "\nChecking the following counters:\n");
	for (num_events = 0, i = 1; i < argc; i++)
	{
		PAPI_event_info_t info;
		int Event;
		char EventName[PAPI_MAX_STR_LEN];
		char *strtoul_check;
		char *counter_last_position = &(argv[i][strlen(argv[i])]);

		/* Check if the user gave us the code or the name */
		Event = strtoul (argv[i], &strtoul_check, 16);
		if (strtoul_check != counter_last_position)
		{
			rc = PAPI_event_name_to_code (argv[i], &Event);
			if (rc != PAPI_OK)
			{
				fprintf (stdout, "Warning! Counter '%s' is not available\n", argv[i]);
				continue;
			}
		}

		/* Get the event name */
		rc = PAPI_event_code_to_name (Event, EventName);
		if (rc != PAPI_OK)
			strcpy (EventName, "unknown");

		/* Get event info,
		   native counters can have info.count == 0 */
		rc = PAPI_get_event_info (Event, &info);
		if (rc != PAPI_OK)
		{
			fprintf (stdout, "Warning! Counter '%s' is not available\n", argv[i]);
			continue;
		}
		else if (info.count == 0 && (Event & PAPI_NATIVE_MASK) == 0)
		{
			fprintf (stdout, "Warning! Counter '%s' is not available\n", argv[i]);
			continue;
		}
		else
		{
			events[num_events++] = Event;
#if 0
			fprintf (stdout, "Counter %s (code %08x): Native? %s (if not, depends on %d native counters)\n", EventName, Event, Event&PAPI_NATIVE_MASK?"yes":"no", info.count);
#endif
			fprintf (stdout, "Counter %s (code %08x): %s", EventName, Event, Event&PAPI_NATIVE_MASK?"native":"derived");
			if (!(Event&PAPI_NATIVE_MASK))
				fprintf (stdout, " (depends on %d native counters)", info.count);
			fprintf (stdout, "\n");
		}
	}

	if (num_events == 0)
	{
		fprintf (stdout, "\n");
		fprintf (stdout, "Sorry, no hardware counters were given\n");
		fprintf (stdout, "Check %s/bin/papi_avail or \n", PAPI_HOME);
		fprintf (stdout, "      %s/bin/papi_native_avail \n", PAPI_HOME);
		fprintf (stdout, "to get a list from the available counters\n");
		return -4;
	}

	fprintf (stdout, "\n");

	CheckInOrder (num_events, events);
	CheckMaxEventSet (num_events, events);

	return 0;
}

