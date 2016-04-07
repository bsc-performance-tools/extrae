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

#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_ASSERT_H
# include <assert.h>
#endif
#if HAVE_LIBGEN_H
# include <libgen.h>
#endif
#if HAVE_MATH_H
# include <math.h>
#endif
#if HAVE_ASSERT_H
# include <assert.h>
#endif

#include <papi.h>

#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <bitset>
#include <algorithm>

using namespace std;

#define MAXBITSET 128 // Increase this if you want to allow more than this value of variable counters

static vector<string> omnipresentCounters, Counters;
static map<string,bool> CounterZero;

static bool miniKernels (void)
{
#define KERNEL_LENGTH  16384
	double d[KERNEL_LENGTH];
	double s[KERNEL_LENGTH];
	double accum[KERNEL_LENGTH];
	unsigned a[KERNEL_LENGTH];
	unsigned u;

	for (u = 0; u < KERNEL_LENGTH; u++)
	{
		a[u] = u;
		d[u] = (double) u;
		s[u] = sqrt (d[u]);
	}

	for (u = 0; u < KERNEL_LENGTH; u++)
		accum[u] = d[u] + a[u];

	for (u = 0; u < KERNEL_LENGTH; u++)
		accum[u] = accum[u]+ (d[u] + a[u]*s[u]);

	assert (a[0] == 0);
	assert (a[KERNEL_LENGTH-1] == KERNEL_LENGTH-1);
	assert (d[0] == 0.);
	assert (d[KERNEL_LENGTH-1] == (double) KERNEL_LENGTH-1);

	memcpy (d, s, KERNEL_LENGTH*sizeof(double));

	return accum[0] == 0.0f+sqrt(0.0f);
}

#if PAPI_VERSION_MAJOR(PAPI_VERSION) < 5
/* Provide our own PAPI_add_named_event if we're on PAPI pre-5 */
static int PAPI_add_named_event (int EventSet, char *counter)
{
	int rc;
	int EventCode;

	if ((rc = PAPI_event_name_to_code (counter, &EventCode)) != PAPI_OK)
		return rc;

	return PAPI_add_event (EventSet, EventCode);
}
#endif

static bool checkCounters (const vector<string> &omnicounters,
	const vector<string> &counters, const bitset<MAXBITSET> &bitmask)
{
	unsigned i;
	bool valid = true;
	int EventSet = PAPI_NULL;
	long long ctrs[32];
	vector<string> CtrsInEventSet;

	if (PAPI_create_eventset(&EventSet) != PAPI_OK)
		return false;

	for (i = 0; i < omnicounters.size() && valid; i++)
	{
		valid = PAPI_add_named_event (EventSet, (char*)omnicounters[i].c_str()) == PAPI_OK;
		CtrsInEventSet.push_back (omnicounters[i]);
	}

	if (!valid)
	{
		cerr << "Error! Omnipresent counters cannot be added together!" << endl;
		exit (-1);
	}

	for (i = 0; i < counters.size() && valid; i++)
		if (bitmask.test(i))
		{
			valid = PAPI_add_named_event (EventSet, (char*)counters[i].c_str()) == PAPI_OK;
			CtrsInEventSet.push_back (counters[i]);
		}

	PAPI_start (EventSet);
	miniKernels();
	PAPI_stop (EventSet, ctrs);

	/* Careful, check whether the counter can be zero or not.
	   We dismiss groups if the counter was not zero in test, but is in this
	   eventset */
	for (i = 0; i < CtrsInEventSet.size(); i++)
		if (CounterZero.count(CtrsInEventSet[i]))
			if (!CounterZero[CtrsInEventSet[i]])
				if (ctrs[i] == 0)
					valid = false;

	PAPI_cleanup_eventset (EventSet);
	PAPI_destroy_eventset (&EventSet);

	return valid;
}

/*   CheckMaxEventSet

     Search the eventset with the maximum events added.
     Uses a bitmask to generate all the possible combinations.
*/
static void CheckMaxEventSet (unsigned neventset, const vector<string> &omnicounters,
	vector<string> &counters)
{
	assert (counters.size() < 63); // We cannot handle values larger than 64 bits!

	bitset<MAXBITSET> bitmask (0);
	bitset<MAXBITSET> max_value (1ULL << counters.size());
	bitset<MAXBITSET> max_combination;

	while (bitmask != max_value)
	{
		unsigned bitvalue = bitmask.to_ulong();
		if (bitmask.count() <= 8-omnicounters.size()) /* Supported per Extrae */
			if (checkCounters (omnicounters, counters, bitmask))
				if (bitmask.count() > max_combination.count())
					max_combination = bitmask;

		/* If reached Extrae's limit, stop here */
		if (max_combination.count() == 8-omnicounters.size())
			break;
#if 0
		string mystring = bitmask.to_string<char,string::traits_type,string::allocator_type>();
		cout << "bits: " << mystring << endl;
#endif
		bitmask = bitvalue+1;
	}
#if 0
	string mystring = max_combination.to_string<char,string::traits_type,string::allocator_type>();
	cout << "max combination bits: " << mystring << endl;
#else

	if (max_combination.count() > 0 || counters.size() == 0)
	{
		/* Show selected counters  */
		cout << "<!-- counter set " << neventset << " -->" << endl;
		cout << "<set enabled=\"yes\" domain=\"all\" changeat-time=\"500000us\">" << endl
		     << "  ";

		// Prepend omnicounters to the max_combination
		if (omnicounters.size() > 0)
		{
			for (size_t i = 0; i < omnicounters.size()-1; i++)
				cout << omnicounters[i] << ",";
			cout << omnicounters[omnicounters.size()-1];
		}

		if (max_combination.count() > 0)
		{
			if (omnicounters.size() > 0)
				cout << ",";

			bitset<MAXBITSET> max = max_combination;
			size_t i = 0;
			while (i < max.size() && max.count() > 0)
			{
				if (max.test(i))
				{
					max.flip (i);
					if (max.count() > 0)
						cout << counters[i] << ",";
					else
						cout << counters[i];
				}
				i++;
			}
		
			/* Remove already used counters from the counter list */
			max = max_combination;
			i = MAXBITSET-1;
			while (1)
			{
				if (max.test(i))
					counters.erase (counters.begin()+i);
				if (i > 0)
					i--;
				else
					break;
			}

		}

		cout << endl << "</set>" << endl;
	}
#endif
}

static void addCounters (const char *ctr, vector<string> & counters)
{
	string s_ctr (ctr);

	size_t position = s_ctr.find (',');
	while (string::npos != position)
	{
		string stmp = s_ctr.substr (0, position);
		if (stmp.length() > 0)
			if (find (counters.begin(), counters.end(), stmp) == counters.end())
				counters.push_back (stmp);
		s_ctr = s_ctr.substr (position+1);
		position  = s_ctr.find (',');
	}
	if (s_ctr.length() > 0)
		if (find (counters.begin(), counters.end(), s_ctr) == counters.end())
			counters.push_back (s_ctr);
}

static unsigned dumpEventCtrInfo (const char *ctr)
{
	PAPI_event_info_t info;
	int Event;
	int rc;
	char EventName[PAPI_MAX_STR_LEN];
	int EventSet = PAPI_NULL;
	long long values;

	rc = PAPI_event_name_to_code ((char*)ctr, &Event);
	if (rc != PAPI_OK)
	{
		cout << "Warning! Counter '" << ctr << "' is not available" << endl;
		return 0;
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
		cout << "Warning! Counter '" << ctr << "' is not available" << endl;
		return 0;
	}
	else if (info.count == 0 && (Event & PAPI_NATIVE_MASK) == 0)
	{
		cout << "Warning! Counter '" << ctr << "' is not available" << endl;
		return 0;
	}
	else
	{
		cout << "Counter " << ctr << " (code " << std::hex << Event << std::dec << "): ";
		if (Event&PAPI_NATIVE_MASK)
			cout << "native";
		else
			cout << "derived";
		if (!(Event&PAPI_NATIVE_MASK))
			cout << " (depends on " << info.count << " native counters)";
	}

	values = 0;
	bool created = false, added = false, started = false;
	if (PAPI_create_eventset(&EventSet) == PAPI_OK)
	{
		created = true;
		if (PAPI_add_named_event (EventSet, (char*)ctr) == PAPI_OK)
		{
			added = true;
			if (PAPI_start (EventSet) == PAPI_OK)
			{
				started = true;
				miniKernels();
				PAPI_stop (EventSet, &values);
				PAPI_cleanup_eventset (EventSet);
				PAPI_destroy_eventset (&EventSet);
			}
		}
	}

	if (!created)
		cout << " -- cannot create eventset!" << endl;
	else if (!added)
		cout << " -- cannot add event!" << endl;
	else if (!started)
		cout << " -- cannot add start eventset!" << endl;
	else if (values == 0)
		cout << " -- warning, 0 value from kernel test!" << endl;
	else if (values > 0)
		cout << endl;

	CounterZero[ctr] = (values == 0);
		
	return 1;
}


int main (int argc, char *argv[])
{
	int rc;
	int i;
	vector<string> omnipresentCounters_tmp, Counters_tmp;

	if (argc < 2)
	{
		cerr << "Usage for " << basename (argv[0]) << endl << endl
		     << "{omnipresent ctr}          ensures that ctr appears in every resulting group" << endl
		     << "{omnipresent ctr1,..,ctrN} ensures that ctr1-ctrN appear in every resulting group" << endl
			 << "ctr                        requests that ctr appears in 1 resulting group" << endl
			 << "ctr1,..,ctrN               requests that ctr1-ctrN appear in 1 resultign group" << endl;

		return -1;
	}

	rc = PAPI_library_init(PAPI_VER_CURRENT);
	if (rc != PAPI_VER_CURRENT && rc > 0)
	{
		cerr << "Error: PAPI library version mismatch!" << endl;
		return -2;
	}

	cout << "This binary was built using PAPI found in " << PAPI_HOME << endl;

	i = 1;
	while (i < argc)
	{
		string param = argv[i];

		if (param == "omnipresent")
		{
			i++;
			if (i < argc)
				addCounters (argv[i], omnipresentCounters_tmp);
		}
		else
		{
			// Add to regular list if it is not already in omnipresent
			if (find (omnipresentCounters_tmp.begin(),
			    omnipresentCounters_tmp.end(),
			    argv[i]) == omnipresentCounters_tmp.end())
				addCounters (argv[i], Counters_tmp);
		}

		i++;
	}

	if (omnipresentCounters.size() >= 8)
	{
		cerr << "Sorry, Extrae is limited to 8 performance counters and you have requested " << omnipresentCounters.size() << " omnipresent counters..." << endl;
		exit (-1);
	}

	vector<string>::iterator it;
	unsigned num_events = 0;
	cout << endl << "** Checking the following omnipresent counters:" << endl;
	for (it = omnipresentCounters_tmp.begin(); it != omnipresentCounters_tmp.end(); it++)
		if (dumpEventCtrInfo ((*it).c_str()))
		{
			omnipresentCounters.push_back (*it);
			num_events++;
		}
	cout << endl << "** Checking the following counters:" << endl;
	for (it = Counters_tmp.begin(); it != Counters_tmp.end(); it++)
		if (dumpEventCtrInfo ((*it).c_str()))
		{
			Counters.push_back (*it);
			num_events++;
		}

	if (num_events == 0)
	{
		cout << endl <<
		     "Sorry, no hardware counters were given or the given are not available." << endl <<
		     "Check " << PAPI_HOME << "/bin/papi_avail or" << endl <<
		     "      " << PAPI_HOME << "/bin/papi_native_avail" << endl <<
		     "to get a list from the available counters in the system." << endl;
		exit (-2);
	}
	else if (num_events >= 64)
	{
		cerr << endl <<
		  "Sorry, we cannot handle 64 or more performance counters at this moment." << endl;
		exit (-3);
	}

	cout << endl;

	size_t ncounters, prevncounters = Counters.size();
	unsigned neventset = 1;
	do
	{
		CheckMaxEventSet (neventset++, omnipresentCounters, Counters);
		ncounters = Counters.size();

		if (prevncounters == ncounters && prevncounters > 0)
		{
			cout << endl <<
			  "Caution, for some reason the following hardware counters cannot be added in an eventset." << endl;
			for (size_t s = 0; s < Counters.size(); s++)
				cout << Counters[s] << " ";
			cout << endl;
			break;
		}
		else
			prevncounters = ncounters;

	} while (Counters.size() > 0);

	return 0;
}

