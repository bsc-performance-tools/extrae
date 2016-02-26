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

#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>

using namespace std; 

#include <BPatch.h>

BPatch *bpatch;
string loadedModule;

int errorPrint = 0; // external "dyninst" tracing (via errorFunc)

#define DYNINST_NO_ERROR -1
int expectError = DYNINST_NO_ERROR;

/*void errorFunc(BPatchErrorLevel level, int num, const char **params) */
void errorFunc(BPatchErrorLevel level, int num, const char* const* params)
{
	if (1 /* verbose */)
	{
		if (num == 0)
		{
			// conditional reporting of warnings and informational messages
			if (level == BPatchInfo)
			{
				if (errorPrint > 1)
					printf("%s\n", params[0]);
			}
			else
				printf("%s", params[0]);
		}
		else
		{
			// reporting of actual errors
			char line[256];
			const char *msg = bpatch->getEnglishErrorString(num);
			bpatch->formatErrorString(line, sizeof(line), msg, params);
	
			if (num != expectError)
			{
				if(num != 112)
					printf ("Error #%d (level %d): %s\n", num, level, line);
			}
		}
	}
}

static BPatch_function * getRoutine (string &routine, BPatch_image *appImage)
{
	BPatch_Vector<BPatch_function *> found_funcs;

	if (appImage->findFunction (routine.c_str(), found_funcs) == NULL)
	{
		string error = string("appImage->findFunction: Failed to find function ")+routine;
		cerr << "WARNING " << error << endl;
		exit (-1);
	}
	if (found_funcs.size() < 1)
	{
		string error = string("appImage->findFunction: Failed to find function ")+routine;
		cerr << "WARNING " << error << endl;
		exit (-1);
	}
	if (found_funcs[0] == NULL)
	{
		string error = string("appImage->findFunction: Failed to find function ")+routine;
		cerr << "WARNING " << error << endl;
		exit (-1);
	}

	return found_funcs[0];
}

int processParams (int argc, char *argv[])
{
	int i = 1;

	if (argc <= 1)
	{
		fprintf (stderr, "pass a binary\n");
		exit (1);
	}

	if (i >= argc)
	{
		fprintf (stderr, "Unable to find the target application\n");
		exit (0);
	}
	
	return i;
}

void Instrumentcall (BPatch_image *appImage, BPatch_process *appProcess)
{
	unsigned insertion = 0;
	unsigned i = 0;

	BPatch_Vector<BPatch_function *> *vfunctions = appImage->getProcedures (true);
	cout << vfunctions->size() << " functions found in binary " << endl;

	cout << "Parsing functions " << flush;

	while (i < vfunctions->size())
	{
		char name[1024], sharedlibname[1024];

		BPatch_function *f = (*vfunctions)[i];
		(f->getModule())->getFullName (sharedlibname, 1024);
		f->getName (name, 1024);

		BPatch_Vector<BPatch_point *> *vpoints = f->findPoint (BPatch_subroutine);

		unsigned j = 0;
		while (j < vpoints->size())
		{
			BPatch_function *called = ((*vpoints)[j])->getCalledFunction();
			if (NULL != called)
			{
				char calledname[1024];
				called->getName (calledname, 1024);
				if (strcmp (calledname, "functionB") == 0)
				{
					string s = "functionA";
					BPatch_function *patch = getRoutine (s, appImage);
					if (patch != NULL)
					{
						bool replaced = appProcess->replaceFunctionCall (*((*vpoints)[j]), *patch);
						if (replaced)
							cout << "call to functionA has been successfully replaced by a call to functionB" << endl;
						else
							cout << "call to functionA failed to replace a call to functionB" << endl;

						insertion++;
					}
				}
			}
			j++;
		}

		i++;
	}

	cout << endl;

	cout << "insertion = " << insertion << endl;
}

int main (int argc, char *argv[], const char *envp[])
{
	int index;
	char *env_var;

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
	env_var = (char*) malloc ((1+strlen("DYNINSTAPI_RT_LIB=")+strlen(DYNINST_RT_LIB))*sizeof(char));

	/* Parse the params */
	index = processParams (argc, argv);

	// Create an instance of the BPatch library
	bpatch = new BPatch;

	// Register a callback function that prints any error messages
	bpatch->registerErrorCallback (errorFunc);

	// Don't check recursion in snippets
	bpatch->setTrampRecursive(true);

	BPatch_process *appProcess = bpatch->processCreate ((const char*) argv[index], (const char**) &argv[index], &envp[0]);
	fprintf (stdout, "* "PACKAGE_NAME": Stopping execution...\n");
	if (!appProcess->stopExecution())
	{
		fprintf (stderr, "* "PACKAGE_NAME": Cannot stop execution of the target application\n");
		exit (-1);
	}

	BPatch_image *appImage = appProcess->getImage();

	cout << "Instrumenting function call " << endl;
	Instrumentcall (appImage, appProcess);

	fprintf (stdout,"* Starting program execution\n");
	if (!appProcess->continueExecution())
	{
		fprintf (stderr, "* "PACKAGE_NAME": cannot continue execution of the target application\n");
		exit (-1);
	}

	while (!appProcess->isTerminated())
		bpatch->waitForStatusChange();

	int retVal;
	if (appProcess->terminationStatus() == ExitedNormally)
		retVal = appProcess->getExitCode();
	else if(appProcess->terminationStatus() == ExitedViaSignal)
		retVal = appProcess->getExitSignal();

	delete appProcess;

	return 0;
}

