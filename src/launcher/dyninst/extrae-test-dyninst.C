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
#include "cpp_utils.h"

#if HAVE_STDLIB_H
# include <stdlib.h>
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
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>

#include "events.h"
#include "mini-xml-parse.h"

using namespace std; 

#include "commonSnippets.h"
#include "applicationType.h"
#include "forkSnippets.h"
#include "cudaSnippets.h"
#include "ompSnippets.h"
#include "mpiSnippets.h"
#include "apiSnippets.h"

#include <BPatch_statement.h>
#include <BPatch_point.h>

static bool ListFunctions = false;
static bool showErrors = true;

static BPatch *bpatch;

vector<BPatch_function *> printfFuncs;

#define DYNINST_NO_ERROR -1

/******************************************************************************
 **      Function name : file_exists (char*)
 **      Author : HSG
 **      Description : Checks whether a file exists
 ******************************************************************************/
static int file_exists (char *fname)
{
#if defined(HAVE_ACCESS)
	return access (fname, F_OK) == 0;
#elif defined(HAVE_STAT64)
	struct stat64 sb;
	stat64 (fname, &sb);
	return (sb.st_mode & S_IFMT) == S_IFREG;
#elif defined(HAVE_STAT)
	struct stat sb;
	stat (fname, &sb);
	return (sb.st_mode & S_IFMT) == S_IFREG;
#else
	int fd = open (fname, O_RDONLY);
	if (fd >= 0)
	{
		close (fd);
		return TRUE;
	}
	else
		return FALSE;
#endif
}

void errorFunc (BPatchErrorLevel level, int num, const char* const* params)
{
	if (showErrors)
	{
		if (num == 0)
		{
			// conditional reporting of warnings and informational messages
			if (level != BPatchInfo)
				cerr << endl << PACKAGE_NAME": " << params[0] << endl;
		}
		else
		{
			// reporting of actual errors
			char line[256];
			const char *msg = bpatch->getEnglishErrorString(num);
			bpatch->formatErrorString(line, sizeof(line), msg, params);
	
			if (num != DYNINST_NO_ERROR)
				if (num != 112)
					cerr << PACKAGE_NAME": Error #" << num << " (level " << level << "): " << line << endl;
		}
	}
}

char * printUsage()
{
    char * buffer = (char*) malloc(1024*sizeof(char*));
    strcat(buffer, "Extrae dyninst utility\n\n");
    strcat(buffer, "Usage: extrae OPTIONS binary\n\n");
    strcat(buffer, "-list-functions       List functions found in the binary/process image.\n");
    strcat(buffer, "-instrument R         Instrument routine R.\n");
    return buffer;
}


static int processParams (int argc, char *argv[], set<string> &routines)
{
	bool leave = false;
	int i = 1;

	if (argc <= 1)
	{
		cerr << PACKAGE_NAME << ": You have to provide a binary to instrument" << endl;
		exit (1);
	}

	while (!leave)
	{
		string s = argv[i];
        if (s == "-h" || s == "-help" || s == "--h" || s == "--help")
        {
            printf("%s\n", printUsage());
            exit(-1);
        }
		else if (s == "-list-functions")
		{
			ListFunctions = true;
			i++;
			leave = (i >= argc);
		}
		else if (s == "-instrument")
		{
			i++;
			if (i < argc)
			{
				routines.insert (argv[i]);
				i++;
			}
			leave = (i >= argc);
		}
		else
			leave = true;
	}
	
	if (i >= argc)
	{
		cerr << "Unable to find the target application\n" << endl;
		exit (0);
	}
	
	return i;
}

static void ShowFunctions (BPatch_image *appImage)
{
	BPatch_Vector<BPatch_function *> *vfunctions = appImage->getProcedures (false);
	cout << PACKAGE_NAME << ": " << vfunctions->size() << " functions found in binary " << endl;

	unsigned i = 0;
	while (i < vfunctions->size())
	{
		char name[1024];
		BPatch_function *f = (*vfunctions)[i];

		f->getName (name, 1024);

		char mname[1024], tname[1024], modname[1024];
		f->getMangledName (mname, 1024);
		f->getTypedName (tname, 1024);
		f->getModuleName (modname, 1024);

		cout << " * " << i+1 << " of " << vfunctions->size() << ", Name: " << name << endl
		     << "    Mangled Name: " << mname << endl
		     << "    Typed Name  : " << tname << endl
		     << "    Module name : " << modname << endl
		     << "    Base address: " << f->getBaseAddr() << endl
		     << "    Instrumentable? " << (f->isInstrumentable()?"yes":"no") << endl
		     << "    In shared library? " << (f->isSharedLib()?"yes":"no") << endl;

		if (f->isSharedLib())
		{
			char sharedlibname[1024];
			BPatch_module *mod = f->getModule();

			mod->getFullName (sharedlibname, 1024);
			cout << "    Full library name: " << sharedlibname << endl;
			
		}
		cout << endl;

		i++;
	} 
}

static bool InstrumentCall (BPatch_image *appImage,
	BPatch_addressSpace *appProcess, const string & r)
{
	vector<BPatch_function *> functions;
	vector<BPatch_point *> *entry, *exit;
	appImage->findFunction (r.c_str(), functions);
	if (functions.size() > 0)
	{
		entry = functions[0]->findPoint(BPatch_entry);
		if (entry != NULL)
		{
			if (entry->size() > 0)
			{
				vector<BPatch_snippet*> printfArgs;
				string s = string(PACKAGE_NAME)+": Entering "+r+"\n";
				BPatch_snippet *strparam = new BPatch_constExpr (s.c_str());
				printfArgs.push_back (strparam);
				BPatch_funcCallExpr printfCall(*(printfFuncs[0]), printfArgs);
				appProcess->insertSnippet (printfCall, *entry);
			}
			else
				return false;
		}
		else
			return false;

		exit = functions[0]->findPoint(BPatch_exit);
		if (exit != NULL)
		{
			if (exit->size() > 0)
			{
				vector<BPatch_snippet*> printfArgs;
				string s = string(PACKAGE_NAME)+": Leaving "+r+"\n";
				BPatch_snippet *strparam = new BPatch_constExpr (s.c_str());
				printfArgs.push_back (strparam);
				BPatch_funcCallExpr printfCall(*(printfFuncs[0]), printfArgs);
				appProcess->insertSnippet (printfCall, *exit);
			}
			else
				return false;
		}
		else
			return false;
	}
	else
		return false;

	return true;
}

static void InstrumentCalls (BPatch_image *appImage,
	BPatch_addressSpace *appProcess, set<string> &USERset)
{
	unsigned UFinsertion = 0;

	cout << PACKAGE_NAME << ": Obtaining functions from application image (this may take a while)..." << flush;
	BPatch_Vector<BPatch_function *> *vfunctions = appImage->getProcedures (false);
	cout << "Done" << endl;

	cout << PACKAGE_NAME << ": Parsing executable looking for instrumentation points (" << vfunctions->size() << ") " << endl;

	if (USERset.size() > 0)
	{
		/* Instrument user functions! */
		set<string>::iterator iter;
		for (iter = USERset.begin(); iter != USERset.end(); iter++)
		{
			if (*iter != "main")
			{
				if (InstrumentCall (appImage, appProcess, *iter))
				{
					UFinsertion++;
					cout << PACKAGE_NAME << ": Instrumenting user function : " << *iter << endl;
				}
			}
			else
				cout << PACKAGE_NAME << ": Although 'main' symbol was in the instrumented functions list, it will not be instrumented" << endl;
		}
		cout << PACKAGE_NAME << ": End of instrumenting functions" << endl;
	}

	/* Protect following lines from reporting errors */
	showErrors = false;

	/* Regular main symbol */
	InstrumentCall (appImage, appProcess, "main"); 

	/* Special cases (e.g., fortran stop call) */
	InstrumentCall (appImage, appProcess, "exit"); /* C */
	InstrumentCall (appImage, appProcess, "_xlfExit"); /* Fortran IBM XL */
	InstrumentCall (appImage, appProcess, "_gfortran_stop_numeric"); /* Fortran GNU */
	InstrumentCall (appImage, appProcess, "for_stop_core"); /* Fortran Intel */

	showErrors = true;
	/* Reenable errors */

	if (USERset.size() > 0)
		cout << PACKAGE_NAME << ": " << UFinsertion << " user function" << ((UFinsertion!=1)?"s":"") << " instrumented" << endl;
}

int main (int argc, char *argv[])
{
	set<string> Routines;
	char *env_var;
	int index;

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

	/* Parse the params */
	index = processParams (argc, argv, Routines);

	/* Does the binary exists? */
	if (!file_exists(argv[index]))
	{
		cout << PACKAGE_NAME << ": Executable " << argv[index] << " cannot be found!" << endl;
		exit (-1);
	}

	if (Routines.size() > 0)
	{
		set<string>::iterator it;
		cout << PACKAGE_NAME << ": Will instrument routines ";
		for (it = Routines.begin(); it != Routines.end(); it++)
			cout << *it << " ";
		cout << endl;
	}

	/* Create an instance of the BPatch library */
	bpatch = new BPatch;

	/* Register a callback function that prints any error messages */
	bpatch->registerErrorCallback (errorFunc);

	/* Don't check recursion in snippets */
	bpatch->setTrampRecursive (true);

	cout << "Welcome to " << PACKAGE_STRING  << " revision " << EXTRAE_SVN_REVISION
	  << " based on " << EXTRAE_SVN_BRANCH << " launcher using DynInst "
	  << DYNINST_MAJOR << "." << DYNINST_MINOR << "." << DYNINST_SUBMINOR << endl;

	int i = 1;
	while (argv[index+i] != NULL)
	{
		cout << PACKAGE_NAME << ": Argument " << i <<  " - " << argv[index+i] << endl;
		i++;
	}

	cout << PACKAGE_NAME << ": Creating process for image binary " << argv[index];
	cout.flush ();
	BPatch_process * appProcess =
	  bpatch->processCreate ((const char*) argv[index], (const char**) &argv[index], (const char**) environ);
	if (appProcess == NULL)
	{
		cerr << endl << PACKAGE_NAME << ": Error creating the target application process" << endl;
		exit (-1);
	}
	cout << endl;

	/* Stop the execution in order to load the instrumentation library */
	cout << PACKAGE_NAME << ": Stopping mutatee execution" << endl;
	if (!appProcess->stopExecution())
	{
		cerr << PACKAGE_NAME << ": Cannot stop execution of the target application" << endl;
		exit (-1);
	}

	cout << PACKAGE_NAME << ": Acquiring process image" << endl;
	BPatch_image *appImage = appProcess->getImage();
	if (appImage == NULL)
	{
		cerr << PACKAGE_NAME << ": Error while acquiring application image" << endl;
		exit (-1);
	}

	cout << PACKAGE_NAME << ": Looking for printf symbol in application image" << endl;
	appImage->findFunction ("printf", printfFuncs);
	if (printfFuncs.size() == 0)
	{
		cerr << PACKAGE_NAME << ": Error! Cannot locate printf function within image" << endl;
		exit (-1);
	}

	if (!ListFunctions)
	{
		InstrumentCalls (appImage, appProcess, Routines);

		cout << PACKAGE_NAME << ": Starting program execution" << endl;
		if (!appProcess->continueExecution())
		{
			/* If the application cannot continue, terminate the mutatee and exit */
			cerr << PACKAGE_NAME << ": Cannot continue execution of the target application" << endl;
			appProcess->terminateExecution();
			exit (-1);
		}

		while (!appProcess->isTerminated())
			bpatch->waitForStatusChange();

		if (appProcess->terminationStatus() == ExitedNormally)
			appProcess->getExitCode();
		else if(appProcess->terminationStatus() == ExitedViaSignal)
			appProcess->getExitSignal();

		delete appProcess;
	}
	else
		ShowFunctions (appImage);

	return 0;
}
