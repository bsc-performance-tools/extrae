/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/launcher/dyninst/apiSnippets.C,v $
 | 
 | @last_commit: $Date: 2008/12/12 16:43:55 $
 | @version:     $Revision: 1.3 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: apiSnippets.C,v 1.3 2008/12/12 16:43:55 harald Exp $";

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

