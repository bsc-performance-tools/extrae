/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  Extrae                                   *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/launcher/dyninst/ompitrace.C,v $
 | 
 | @last_commit: $Date$
 | @version:     $Revision$
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id$";

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

static int file_exists (char *fitxer)
{
  struct stat buffer;
  return stat(fitxer, &buffer)== 0;
}

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

int main (int argc, char *argv[], const char *envp[])
{
	if (getenv("EXTRAE_HOME") == NULL)
	{
		cerr << PACKAGE_NAME << ": Environment variable EXTRAE_HOME is undefined" << endl;
		exit (-1);
	}

	if (argc != 3)
	{
		cout << "Options: binary library" << endl;
		exit (-1);
	}

	/* Does the binary exists? */
	if (!file_exists(argv[1]))
	{
		cout << PACKAGE_NAME << ": Executable " << argv[1] << " cannot be found!" << endl;
		exit (-1);
	}

	/* Create an instance of the BPatch library */
	bpatch = new BPatch;

	/* Register a callback function that prints any error messages */
	bpatch->registerErrorCallback (error_function);

	/* Don't check recursion in snippets */
	bpatch->setTrampRecursive (true);

	cout << PACKAGE_NAME << ": Creating process for image binary " << argv[1] << endl;

	BPatch_process *appProcess = bpatch->processCreate ((const char*) argv[1], (const char**) NULL, (const char**) environ);

	/* Stop the execution in order to load the instrumentation library */
	cout << PACKAGE_NAME << ": Stopping mutatee execution" << endl;
	if (!appProcess->stopExecution())
	{
		cerr << PACKAGE_NAME << ": Cannot stop execution of the target application" << endl;
		exit (-1);
	}

	if (!file_exists (argv[2]))
	{
		/* If the library does not exist, terminate the mutatee and exit */
		cerr << PACKAGE_NAME << ": Cannot find the library " << argv[2] << endl;
		appProcess->terminateExecution();
		exit (-1);
	}
	if (!appProcess->loadLibrary (argv[2]))
		cout << PACKAGE_NAME << ": Cannot load library " << argv[2] << endl;
	else
		cout << PACKAGE_NAME << ": Can load library " << argv[2] << endl;

	appProcess->terminateExecution();

	delete appProcess;

	return 0;
}

