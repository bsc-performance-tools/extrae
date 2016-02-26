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

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_UNISTD_H
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
	const char *name;
	BPatch_function *patch;
};

#define APIROUTINE_T_S(x)  {#x,NULL}
#define APIROUTINE_C_T(x)  APIROUTINE_T_S(OMPtrace##x), APIROUTINE_T_S(MPItrace##x), APIROUTINE_T_S(OMPItrace##x), APIROUTINE_T_S(SEQtrace##x), APIROUTINE_T_S(Extrae##x)
#define APIROUTINE_F_T(x)  APIROUTINE_T_S(omptrace##x), APIROUTINE_T_S(mpitrace##x), APIROUTINE_T_S(ompitrace##x), APIROUTINE_T_S(seqtrace##x), APIROUTINE_T_S(extrae##x)
#define APIROUTINE_T(x)    APIROUTINE_C_T(x), APIROUTINE_F_T(x)
#define APIROUTINE_T_END   {NULL,NULL}

static struct APIroutines_t APIroutines[] =
	{
		APIROUTINE_T(_init),
		APIROUTINE_T(_fini),
		APIROUTINE_T(_is_initialized),
		APIROUTINE_T(_event),
		APIROUTINE_T(_nevent),
		APIROUTINE_T(_eventandcounters),
		APIROUTINE_T(_neventandcounters),
		APIROUTINE_T(_counters),
		APIROUTINE_T(_shutdown),
		APIROUTINE_T(_restart),
		APIROUTINE_T(_next_hwc_set),
		APIROUTINE_T(_previous_hwc_set),
		APIROUTINE_T(_define_event_type),
		APIROUTINE_T_END
	};


void loadAPIPatches (BPatch_image *appImage)
{
	int i;

	cout << PACKAGE_NAME << ": Loading instrumentation API patches..." << flush;

	i = 0;
	while (APIroutines[i].name != NULL)
	{
		APIroutines[i].patch = getRoutine (APIroutines[i].name, appImage);
		if (NULL == APIroutines[i].patch)
			cerr << "Unable to find " << APIroutines[i].name << " in the application image" << endl;
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

