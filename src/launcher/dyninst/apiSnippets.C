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
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <list>
#include <string>
#include <iostream>
#include <fstream>

#include "commonSnippets.h"

#include <BPatch_function.h>

using namespace std; 

struct APIroutines_t
{
	char *name;
	BPatch_function *patch;
};

#define APIROUTINE_T_S(x)  {#x,NULL}
#define APIROUTINE_C_T(x)  APIROUTINE_T_S(OMP##x), APIROUTINE_T_S(MPI##x), APIROUTINE_T_S(OMPI##x), APIROUTINE_T_S(SEQ##x)
#define APIROUTINE_F_T(x)  APIROUTINE_T_S(omp##x), APIROUTINE_T_S(mpi##x), APIROUTINE_T_S(ompi##x), APIROUTINE_T_S(seq##x)
#define APIROUTINE_T(x)    APIROUTINE_C_T(x), APIROUTINE_F_T(x)
#define APIROUTINE_T_END   {NULL,NULL}

static struct APIroutines_t APIroutines[] =
	{
		APIROUTINE_T(trace_init),
		APIROUTINE_T(trace_fini),
		APIROUTINE_T(trace_event),
		APIROUTINE_T(trace_nevent),
		APIROUTINE_T(trace_eventandcounters),
		APIROUTINE_T(trace_neventandcounters),
		APIROUTINE_T(trace_counters),
		APIROUTINE_T(trace_shutdown),
		APIROUTINE_T(trace_restart),
		APIROUTINE_T(trace_next_hwc_set),
		APIROUTINE_T(trace_previous_hwc_set),
		
		APIROUTINE_T_END
	};


void loadAPIPatches (BPatch_image *appImage)
{
	int i;

	cout << PACKAGE_NAME << ": Loading instrumentation API patches..." << flush;

	i = 0;
	while (APIroutines[i].name != NULL)
	{
		string s = APIroutines[i].name;
		APIroutines[i].patch = getRoutine (s, appImage);
		if (NULL == APIroutines[i].patch)
			cerr << "Unable to find " << s << " in the application image" << endl;
		i++;
	}

	cout << "Done" << endl;
}

BPatch_function * getAPIPatch (char *routine)
{
	BPatch_function *res = NULL;
	int i;

	i = 0;
	while (APIroutines[i].name != NULL)
	{
		if (!strcmp(APIroutines[i].name, routine))
		{
			res = APIroutines[i].patch;
			break;
		}
		i++;
	}
	return res;
}

