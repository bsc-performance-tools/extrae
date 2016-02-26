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
#if HAVE_LIBGEN_H
# include <libgen.h>
#endif

#include <sys/stat.h>

#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std; 

#include <BPatch.h>

BPatch *bpatch;

void error_function (BPatchErrorLevel level, int num, const char* const* params)
{
	if (num == 0)
	{
		if (level == BPatchInfo)
			fprintf (stderr, "%s\n", params[0]);
		else
			fprintf (stderr, "%s", params[0]);
	}
	else
	{
		char line[256];
		const char *msg = bpatch->getEnglishErrorString(num);
		bpatch->formatErrorString(line, sizeof(line), msg, params);

		if (num != -1)
			if (num != 112)
				fprintf (stderr, "Error #%d (level %d): %s\n", num, level, line);
	}
}

static void ForkCallback (BPatch_thread *parent, BPatch_thread *child)
{
	if (child == NULL)
	{
		/* preFork */
	}
	else
	{
		/* postFork */

		parent->oneTimeCodeAsync (BPatch_nullExpr());

//		BPatch_process *p = parent->getProcess();
//		p->stopExecution();
//		p->oneTimeCode (BPatch_nullExpr());
//		p->continueExecution();
	}
}

int main (int argc, char *argv[])
{
	char *env_var;

	if (argc != 2)
	{
		cout << "Options: binary" << endl;
		exit (-1);
	}

	if ((env_var = getenv ("DYNINSTAPI_RT_LIB")) == NULL)
	{
		env_var = (char*) malloc ((1+strlen("DYNINSTAPI_RT_LIB=")+strlen(DYNINST_RT_LIB))*sizeof(char));
		if (env_var == NULL)
		{
			cerr << PACKAGE_NAME << ": Cannot allocate memory to define DYNINSTAPI_RT_LIB!" << endl;
			exit (-1);
		}
		sprintf (env_var, "DYNINSTAPI_RT_LIB=%s", DYNINST_RT_LIB);
		putenv (env_var);
	}
	else
		cout << PACKAGE_NAME << ": Warning, DYNINSTAPI_RT_LIB already set and pointing to " << 
		  env_var << endl;

	/* Create an instance of the BPatch library */
	bpatch = new BPatch;

	/* Register a callback function that prints any error messages */
	bpatch->registerErrorCallback (error_function);

	/* Don't check recursion in snippets */
	bpatch->setTrampRecursive (true);

	cout << PACKAGE_NAME << ": Creating process for image binary " << argv[1] << endl;

	BPatch_process *appProcess = bpatch->processCreate ((const char*) argv[1], (const char**) NULL, (const char**) environ);

	bpatch->registerPreForkCallback (ForkCallback);
	bpatch->registerPostForkCallback (ForkCallback);

	if (!appProcess->continueExecution())
	{
		/* If the application cannot continue, terminate the mutatee and exit */
		cerr << PACKAGE_NAME << ": Cannot continue execution of the target application" << endl;
		appProcess->terminateExecution();
		exit (-1);
	}

	while (!appProcess->isTerminated())
		bpatch->waitForStatusChange();

	return 0;
}

