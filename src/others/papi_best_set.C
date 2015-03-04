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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/trunk/src/others/papi_best_set.c $
 | @last_commit: $Date: 2011-11-30 11:58:56 +0100 (mié, 30 nov 2011) $
 | @version:     $Revision: 890 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: papi_best_set.c 890 2011-11-30 10:58:56Z harald $";

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

#include <vector>
#include <string>
#include <iostream>
#include <bitset>

using namespace std;

#define MAXBITSET 128 // Increase this if you want to allow more than this value of variable counters

static vector<string> omnipresentCounters, Counters;

bool checkCounters (const vector<string> &omnicounters,
	const vector<string> &counters, const bitset<MAXBITSET> &bitmask)
{
	unsigned i;
	bool valid = true;
	int EventSet = PAPI_NULL;

	if (PAPI_create_eventset(&EventSet) != PAPI_OK)
		return false;

	for (i = 0; i < omnicounters.size() && valid; i++)
		valid = PAPI_add_named_event (EventSet, (char*)omnicounters[i].c_str()) == PAPI_OK;

	if (!valid)
	{
		cerr << "Error! Omnipresent counters cannot be added together!" << endl;
		exit (-1);
	}

	for (i = 0; i < counters.size() && valid; i++)
		if (bitmask.test(i))
			valid = PAPI_add_named_event (EventSet, (char*)counters[i].c_str()) == PAPI_OK;

	PAPI_cleanup_eventset (EventSet);
	PAPI_destroy_eventset (&EventSet);

	return valid;
}

/*   CheckMaxEventSet

     Search the eventset with the maximum events added.
     Uses a bitmask to generate all the possible combinations.
*/
void CheckMaxEventSet (unsigned neventset, const vector<string> &omnicounters,
	vector<string> &counters)
{
	bitset<MAXBITSET> bitmask (0);
	bitset<MAXBITSET> max_value ( 1 << counters.size() );
	bitset<MAXBITSET> max_combination;

	while (bitmask != max_value)
	{
		unsigned bitvalue = bitmask.to_ulong();
		if (checkCounters (omnicounters, counters, bitmask))
			if (bitmask.count() > max_combination.count())
				max_combination = bitmask;
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

	/* Show selected counters  */
	cout << "<!-- counter set " << neventset << " -->" << endl;
	cout << "<set enabled=\"yes\" domain=\"all\" changeat-time=\"500000us\">" << endl
	     << "  ";

	for (size_t i = 0; i < omnicounters.size()-1; i++)
		cout << omnicounters[i] << ",";
	cout << omnicounters[omnicounters.size()-1];
	
	if (max_combination.count() > 0)
	{
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
#endif
}



void addCounters (char *ctr, vector<string> & counters)
{
	string s_ctr (ctr);

	size_t position = s_ctr.find (',');
	while (string::npos != position)
	{
		counters.push_back (s_ctr.substr (0, position));
		s_ctr = s_ctr.substr (position+1);
		position  = s_ctr.find (',');
	}
	counters.push_back (s_ctr);
}

unsigned dumpEventCtrInfo (const char *ctr)
{
	PAPI_event_info_t info;
	int Event;
	int rc;
	char EventName[PAPI_MAX_STR_LEN];

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
		cout << endl;
	}
	return 1;
}


int main (int argc, char *argv[])
{
	int rc;
	int i;

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
				addCounters (argv[i], omnipresentCounters);
		}
		else
			addCounters (argv[i], Counters);

		i++;
	}

	vector<string>::iterator it;
	unsigned num_events = 0;
	cout << endl << "** Checking the following omnipresent counters:" << endl;
	for (it = omnipresentCounters.begin(); it != omnipresentCounters.end(); it++)
		num_events += dumpEventCtrInfo ((*it).c_str());
	cout << endl << "** Checking the following counters:" << endl;
	for (it = Counters.begin(); it != Counters.end(); it++)
		num_events += dumpEventCtrInfo ((*it).c_str());

	if (num_events == 0)
	{
		cout << endl <<
		     "Sorry, no hardware counters were given or the given are not available." << endl <<
		     "Check " << PAPI_HOME << "/bin/papi_avail or" << endl <<
		     "      " << PAPI_HOME << "/bin/papi_native_avail" << endl <<
		     "to get a list from the available counters in the system." << endl;
		exit (-2);
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
				cout << Counters[i] << " ";
			cout << endl;
			break;
		}
		else
			prevncounters = ncounters;

	} while (Counters.size() > 0);

	return 0;
}

