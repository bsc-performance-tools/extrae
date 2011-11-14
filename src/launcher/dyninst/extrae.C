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

#include "events.h"
#include "mini-xml-parse.h"

using namespace std; 

#include "commonSnippets.h"
#include "applicationType.h"
#include "ompSnippets.h"
#include "mpiSnippets.h"
#include "apiSnippets.h"

#include <BPatch_statement.h>

unsigned int OMPNamingConvention = 0;

BPatch *bpatch;

char *excludeUF = NULL; /* Fitxer que indica quines User Functions s'exclouen */
char *includeUF = NULL; /* Fitxer que indica quines User Functions s'inclouen */
char *excludePF = NULL; /* Fitxer que indica quines Par Functions s'exclouen */

char *configXML = NULL; /* XML configuration file */

int VerboseLevel = 0;  /* Verbose Level */
bool useHWC = false;
bool ListFunctions = false;

string loadedModule;

int errorPrint = 0; // external "dyninst" tracing (via errorFunc)

#define DYNINST_NO_ERROR -1

/******************************************************************************
 **      Function name : file_exists (char*)
 **      Author : HSG
 **      Description : Checks whether a file exists
 ******************************************************************************/
static int file_exists (char *fitxer)
{
  struct stat buffer;
  return stat(fitxer, &buffer)== 0;
}

/*void errorFunc(BPatchErrorLevel level, int num, const char **params) */
void errorFunc(BPatchErrorLevel level, int num, const char* const* params)
{
	if (VerboseLevel)
	{
		if (num == 0)
		{
			// conditional reporting of warnings and informational messages
			if (level == BPatchInfo)
			{
				if (errorPrint > 1)
					fprintf (stderr, "%s\n", params[0]);
			}
			else
			{
				fprintf (stderr, "%s", params[0]);
			}
		}
		else
		{
			// reporting of actual errors
			char line[256];
			const char *msg = bpatch->getEnglishErrorString(num);
			bpatch->formatErrorString(line, sizeof(line), msg, params);
	
			if (num != DYNINST_NO_ERROR)
			{
				if (num != 112)
					fprintf (stderr, "Error #%d (level %d): %s\n", num, level, line);
			}
		}
	}
}

static BPatch_function * getFunction (BPatch_image *appImage, string &name)
{
	BPatch_Vector<BPatch_function *> found_funcs;

	if (appImage->findFunction (name.c_str(), found_funcs) == NULL)
		return NULL;

	if (found_funcs.size() < 1)
		return NULL;

	return found_funcs[0];
}

static void GenerateSymFile (list<string> &ParFunc, list<string> &UserFunc, BPatch_image *appImage, BPatch_process *appProces)
{
  ofstream symfile;
	string symname = string(::XML_GetFinalDirectory())+string("/")+string(::XML_GetTracePrefix())+".sym";

  symfile.open (symname.c_str());
	if (!symfile.good())
	{
		cerr << "Cannot create the symbolic file" << symname << endl;
		return;
	}

	for (list<string>::iterator iter = ParFunc.begin();
		iter != ParFunc.end(); iter++)
	{
		BPatch_function *f = getFunction (appImage, *iter);

		if (f != NULL)
		{
			BPatch_Vector< BPatch_statement > lines;

			appProces->getSourceLines ((unsigned long) f->getBaseAddr(), lines);
			if (lines.size() > 0)
			{
				symfile << "P " << hex << f->getBaseAddr() <<  dec << " " << *iter << " " <<  lines[0].fileName() <<  " " << lines[0].lineNumber() << endl;
			}
			else
			{
				/* this happens if the application was not compiled with -g */
				char modname[1024];
				f->getModuleName (modname, 1024);
				symfile << "P " << hex << f->getBaseAddr() <<  dec << " " << *iter << " " << modname << " 0" << endl;
			}
		}
	}

	for (list<string>::iterator iter = UserFunc.begin();
		iter != UserFunc.end(); iter++)
	{
		BPatch_function *f = getFunction (appImage, *iter);

		if (f != NULL)
		{
			BPatch_Vector< BPatch_statement > lines;

			appProces->getSourceLines ((unsigned long) f->getBaseAddr(), lines);
			if (lines.size() > 0)
			{
				symfile << "U " << hex << f->getBaseAddr() <<  dec << " " << *iter << " " <<  lines[0].fileName() <<  " " << lines[0].lineNumber() << endl;
			}
			else
			{
				/* this happens if the application was not compiled with -g */
				char modname[1024];
				f->getModuleName (modname, 1024);
				symfile << "U " << hex << f->getBaseAddr() <<  dec << " " << *iter << " " << modname << " 0" << endl;
			}
		}
	}

  symfile.close();
}

static int processParams (int argc, char *argv[])
{
	bool sortir = false;
	int i = 1;

	if (argc <= 1)
	{
		cerr << PACKAGE_NAME << ": You have to provide a binary to instrument" << endl;
		exit (1);
	}

	while (!sortir)
	{
		if (strcmp (argv[i], "-exclude") == 0)
		{
			i++;
			excludeUF = argv[i];
			i++;
			sortir = (i >= argc);
		}
		else if (strcmp (argv[i], "-include") == 0)
		{
			i++;
			includeUF = argv[i];
			i++;
			sortir = (i >= argc);
		}
#if defined(ALLOW_EXCLUDE_PARALLEL)
		else if (strcmp (argv[i], "-exclude-parallel") == 0)
		{
			i++;
			excludePF = argv[i];
			i++;
			sortir = (i >= argc);
		}
#endif
		else if (strcmp (argv[i], "-config") == 0)
		{
			i++;
			configXML = argv[i];
			i++;
			sortir = (i >= argc);
		}
		else if (strcmp(argv[i], "-counters") == 0)
		{
			useHWC = true;
			i++;
			sortir = (i >= argc);
		}
		else if (strcmp (argv[i], "-v") == 0)
		{
			VerboseLevel++;
			i++;
			sortir = (i >= argc);
		}
		else if (strcmp (argv[i], "-list-functions") == 0)
		{
			ListFunctions = true;
			i++;
			sortir = (i >= argc);
		}
		else
		{
			sortir = true;
		}
	}
	
	if (i >= argc)
	{
		cerr << "Unable to find the target application\n" << endl;
		exit (0);
	}
	
	return i;
}

static void ReadFileIntoList (char *fitxer, list<string>& container)
{
  char str[2048];

  fstream file_op (fitxer, ios::in);

	if (!file_op.good())
	{
		cerr << PACKAGE_NAME << "Error: Cannot open file " << fitxer << endl;
		return;
	}

	if (VerboseLevel >= 2)
		cout << PACKAGE_NAME << ": Read";

	while (file_op >> str)
	{
		/* Add element into list if it didn't exist */
		list<string>::iterator iter = find (container.begin(), container.end(), str);

		if (iter == container.end())
		{
			if (VerboseLevel >= 2)
				cout << " " << str;
			container.push_back (str);
		}
	}

	if (VerboseLevel >= 2)
		cout << endl;

	file_op.close();
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

		if (VerboseLevel)
		{
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
		}
		else
		{
			cout << name << endl;
		}

		i++;
	} 
}

static void printCallingSites (int id, int total, char *name, string sharedlibname, BPatch_Vector<BPatch_point *> *vpoints)
{
	if (vpoints->size() > 0)
	{
		unsigned j = 0;
		cout << id << " of " << total << " - Calling sites for " << name << " within " << sharedlibname << " (num = " << vpoints->size() << "):" << endl;
		while (j < vpoints->size())
		{
			BPatch_function *called = ((*vpoints)[j])->getCalledFunction();
			if (NULL != called)
			{
				char calledname[1024];
				called->getName (calledname, 1024);
				cout << j+1 << " Calling " << calledname;
				BPatch_procedureLocation loc = ((*vpoints)[j])->getPointType();
				if (loc == BPatch_entry)
					cout << " (entry)" << endl;
				else if (loc == BPatch_exit)
					cout << " (exit)" << endl;
				else if (loc == BPatch_subroutine)
					cout << " (subroutine)" << endl;
				else
					cout << " (unknown point type)" << endl;
			}
			else
			{
				cout << j+1 << " Undetermined " << endl;
			}
			j++;
		}
		cout << endl;
	}
}

static void InstrumentCalls (BPatch_image *appImage, BPatch_process *appProcess,
	ApplicationType *appType, list<string> &OMPList, list<string> &UserList,
	bool instrumentMPI, bool instrumentOMP, bool instrumentUF)
{
	unsigned i = 0;
	unsigned OMPinsertion = 0;
	unsigned MPIinsertion = 0;
	unsigned APIinsertion = 0;
	unsigned UFinsertion = 0;
	const char *PMPI_C_prefix = "PMPI_";
	const char *PMPI_F_prefix = "pmpi_";
	const char *MPI_C_prefix = "MPI_";
	const char *MPI_F_prefix= "mpi_";

	cout << PACKAGE_NAME << ": Obtaining functions from application image (this may take a while)..." << flush;
	BPatch_Vector<BPatch_function *> *vfunctions = appImage->getProcedures (false);
	cout << "Done" << endl;

	cout << PACKAGE_NAME << ": Parsing executable looking for instrumentation points (" << vfunctions->size() << ") ";
	if (VerboseLevel)
		cout << endl;
	else
		cout << flush;

	/*
	  The 1st step includes:
	  a) gather information of openmp outlined routines (original is added to UserList),
	  b) instrument openmp outlined routines
	  c) instrument mpi calls
	  d) instrument api calls
	*/

	while (i < vfunctions->size())
	{
		char name[1024], sharedlibname_c[1024];

		BPatch_function *f = (*vfunctions)[i];
		(f->getModule())->getFullName (sharedlibname_c, 1024);
		f->getName (name, 1024);

		string sharedlibname = sharedlibname_c;

		string sharedlibname_ext;
		if (sharedlibname.rfind('.') != string::npos)
			sharedlibname_ext = sharedlibname.substr (sharedlibname.rfind('.'));
		else
			sharedlibname_ext = "";

#if 1
		if (instrumentOMP && appType->get_isOpenMP() && loadedModule != sharedlibname)
		{
			/* OpenMP instrumentation (just for OpenMP apps) */
			if (appType->isMangledOpenMProutine (name))
			{
				if (VerboseLevel)
					cout << PACKAGE_NAME << ": Instrumenting OpenMP outlined routine " << name << endl;

				/* Instrument routine */ 
				wrapTypeRoutine (f, name, OMPFUNC_EV, appImage, appProcess);

				/* Add to list if not already there */
				OMPList.push_back (name);

				/* Demangle name and add into the UF list if it didn't exist there */
				string demangled = appType->demangleOpenMProutine (name);

				list<string>::iterator iter = find (UserList.begin(), UserList.end(), demangled);
				if (iter == UserList.end())
				{
					if (demangled != "main")
					{
						UserList.push_back (demangled);

						if (VerboseLevel)
							cout << PACKAGE_NAME << ": Adding demangled OpenMP routine " << demangled << " to the user function list" << endl;	
					}
					else
					{
						if (VerboseLevel)
							cout << PACKAGE_NAME << ": will not add main as a demangled OpenMP routine" << endl;
					}
				}

				OMPinsertion++;
			}
		}
#endif
		if (sharedlibname_ext == ".f" || sharedlibname_ext == ".F" || /* fortran */
		  sharedlibname_ext == ".for" || sharedlibname_ext == ".FOR" || /* fortran */
		  sharedlibname_ext == ".f90" || sharedlibname_ext == ".F90" || /* fortran 90 */
		  sharedlibname_ext == ".f77" || sharedlibname_ext == ".F77" || /* fortran 77 */
		  sharedlibname_ext == ".c" || sharedlibname_ext == ".C" || /* C */
		  sharedlibname_ext == ".cxx" || sharedlibname_ext == ".cpp" || /* c++ */
		  sharedlibname_ext == ".c++" || /* c++ */
		  sharedlibname_ext == ".i" || /* some compilers generate this extension in intermediate files */ 
		  sharedlibname == "DEFAULT_MODULE" /* Dyninst specific container that represents the executable */
	  )
		{
			/* API instrumentation (for any kind of apps)
	
			   Skip calls from my own module
			*/
			BPatch_Vector<BPatch_point *> *vpoints = f->findPoint (BPatch_subroutine);

			if (vpoints == NULL)
				break;

			if (VerboseLevel >= 2)
				printCallingSites (i, vfunctions->size(), name, sharedlibname, vpoints);

			unsigned j = 0;
			while (j < vpoints->size())
			{
				BPatch_function *called = ((*vpoints)[j])->getCalledFunction();
				if (NULL != called)
				{
					char calledname[1024];
					called->getName (calledname, 1024);

					/* Check API calls */
					BPatch_function *patch_api = getAPIPatch (calledname);
					if (patch_api != NULL)
					{
						if (appProcess->replaceFunctionCall (*((*vpoints)[j]), *patch_api))
						{
							APIinsertion++;
							if (VerboseLevel)
								cout << PACKAGE_NAME << ": Replaced call " << calledname << " in routine " << name << "(" << sharedlibname << ")" << endl;
						}
						else
							cerr << PACKAGE_NAME << ": Cannot replace " << calledname << " routine" << endl;
					}

					/* Check MPI calls */
					if (instrumentMPI && appType->get_isMPI() && (
					    strncmp (calledname, PMPI_C_prefix, 5) == 0 || strncmp (calledname, MPI_C_prefix, 4) == 0 ||
					    strncmp (calledname, PMPI_F_prefix, 5) == 0 || strncmp (calledname, MPI_F_prefix, 4) == 0))
					{
						BPatch_function *patch_mpi = getMPIPatch (calledname);
						if (patch_mpi != NULL)
						{
							if (appProcess->replaceFunctionCall (*((*vpoints)[j]), *patch_mpi))
							{
								MPIinsertion++;
								if (VerboseLevel)
									cout << PACKAGE_NAME << ": Replaced call " << calledname << " in routine " << name << " (" << sharedlibname << ")" << endl;
							}
							else
								cerr << PACKAGE_NAME << ": Cannot replace " << calledname << " routine" << endl;
						}
					}
				}
				j++;
			}
		}

		i++;

		if (!VerboseLevel)
		{
			if (i == 1)
				cout << "1" << flush;
			else if (i%1000 == 0)
				cout << i << flush;
			else if (i%100 == 0)
				cout << "." << flush;
		}
	}

	if (!VerboseLevel)
		cout << ".Done" << endl;

	if (UserList.size() > 0 && instrumentUF)
	{
		/* Instrument user functions! */

		cout << PACKAGE_NAME << ": Instrumenting user functions...";
		if (VerboseLevel)
			cout << endl;
		else
			cout << flush;

		list<string>::iterator iter = UserList.begin();
		while (iter != UserList.end())
		{
			BPatch_function *f = getFunction (appImage, *iter);

			if (f != NULL)
			{
				wrapTypeRoutine (f, *iter, USRFUNC_EV, appImage, appProcess);
				UFinsertion++;

				if (VerboseLevel)
					cout << PACKAGE_NAME << ": Instrumenting user function : " << *iter << endl;
			}
			else
			{
				if (VerboseLevel)
					cout << PACKAGE_NAME << ": Unable to instrument user function : " << *iter << endl;
			}
			iter++;
		}
		if (VerboseLevel)
			cout << PACKAGE_NAME << ": End of instrumenting functions" << endl;
		else
			cout << "Done" << endl;
	}

	cout << PACKAGE_NAME << ": " << APIinsertion << " API patches applied" << endl;
	if (appType->get_isMPI())
		cout << PACKAGE_NAME << ": " << MPIinsertion << " MPI patches applied" << endl;
	if (appType->get_isOpenMP())
		cout << PACKAGE_NAME << ": " << OMPinsertion << " OpenMP patches applied to outlined routines" << endl;
	if (UserList.size() > 0)
		cout << PACKAGE_NAME << ": " << UFinsertion << " user function" << ((UFinsertion!=1)?"s":"") << " instrumented" << endl;
}

int main (int argc, char *argv[])
{
	list<string> UserFunctions;
	list<string> ParallelFunctions;
	list<string> excludedUserFunctions;
#if defined(ALLOW_EXCLUDE_PARALLEL)
	list<string> excludedParallelFunctions;
#endif

	int index;

	if (getenv("EXTRAE_HOME") == NULL)
	{
		cerr << PACKAGE_NAME << ": Environment variable EXTRAE_HOME is undefined" << endl;
		exit (-1);
	}

#if 0
	if (getenv("DYNINSTAPI_RT_LIB") == NULL)
	{
		/* DYNINSTAPI_RT_LIB=%s/lib/libdyninstAPI_RT.so.1 is +/- 50 chars */
		char *env = (char*) malloc((strlen(getenv("EXTRAE_HOME"))+50)*sizeof(char));
		sprintf (env, "DYNINSTAPI_RT_LIB=%s/lib/libdyninstAPI_RT.so.1",  getenv("EXTRAE_HOME"));
		putenv (env);
		fprintf (stderr, "Environment variable DYNINSTAPI_RT_LIB is undefined.\nUsing ${EXTRAE_HOME}/lib/libdyninstAPI_RT.so.1 instead\n");
	}
#endif

	/* Parse the params */
	index = processParams (argc, argv);

	if (!ListFunctions)
	{
		if (configXML != NULL)
		{
			char * envvar = (char *) malloc ((strlen(configXML)+strlen("EXTRAE_CONFIG_FILE=")+1)*sizeof (char));
			if (NULL == envvar)
			{
				cerr << PACKAGE_NAME << ": Error! Unable to allocate memory for EXTRAE_CONFIG_FILE environment variable" << endl;
				exit (-1);
			}
			sprintf (envvar, "EXTRAE_CONFIG_FILE=%s", configXML);
			putenv (envvar);
		}
		else
		{
			if (getenv ("EXTRAE_CONFIG_FILE") == NULL)
			{
				cerr << PACKAGE_NAME << ": Error! You have to provide a configuration file using the -config parameter or set the EXTRAE_CONFIG_FILE" << endl;
				exit (-1);
			}
			configXML = getenv ("EXTRAE_CONFIG_FILE");
		}

		if (!file_exists(configXML))
		{
			cerr << PACKAGE_NAME << ": Error! Unable to locate " << configXML << endl;	
			exit (-1);
		}

		::Parse_XML_File (0, 0, configXML);
	}

	/* Does the binary exists? */
	if (!file_exists(argv[index]))
	{
		cout << PACKAGE_NAME << ": Executable " << argv[index] << " cannot be found!" << endl;
		exit (-1);
	}

	/* Create an instance of the BPatch library */
	bpatch = new BPatch;

	/* Register a callback function that prints any error messages */
	bpatch->registerErrorCallback (errorFunc);

	/* Don't check recursion in snippets */
	bpatch->setTrampRecursive (true);

	cout << PACKAGE_NAME << ": Creating process for image binary " << argv[index] << endl;
	int i = 1;
	while (argv[index+i] != NULL)
	{
		cout << PACKAGE_NAME << ": Argument " << i <<  " - " << argv[index+i] << endl;
		i++;
	}

	char buffer[1024];
	BPatch_process *appProcess = bpatch->processCreate ((const char*) argv[index], (const char**) &argv[index], (const char**) environ);

	/* Stop the execution in order to load the instrumentation library */
	cout << PACKAGE_NAME << ": Stopping mutatee execution" << endl;
	if (!appProcess->stopExecution())
	{
		cerr << PACKAGE_NAME << ": Cannot stop execution of the target application" << endl;
		exit (-1);
	}

	BPatch_image *appImage = appProcess->getImage();

	/* The user asks for the list of functions, simply show it */
	if (ListFunctions)
	{
		ShowFunctions (appImage);
		appProcess->terminateExecution();
		exit (-1);
	}

	/* Read files */
	if (::XML_have_UFlist())
	{
		if (VerboseLevel)
			cout << PACKAGE_NAME << ": Reading instrumented user functions from " << ::XML_UFlist() << endl;
		ReadFileIntoList (::XML_UFlist(), UserFunctions);
	}

	if (::XML_CheckTraceEnabled())
	{
		ApplicationType *appType = new ApplicationType ();
		appType->detectApplicationType (appImage);
		appType->dumpApplicationType ();

		/* Check for the correct library to be loaded */
		if (appType->get_isMPI())
		{
			if (appType->get_isOpenMP())
				sprintf (buffer, "%s/lib/lib_dyn_ompitrace.so", getenv("EXTRAE_HOME"));
			else
#if defined(MPI_COMBINED_C_FORTRAN)
				sprintf (buffer, "%s/lib/lib_dyn_mpitrace.so", getenv("EXTRAE_HOME"));
#else
				sprintf (buffer, "%s/lib/lib_dyn_mpitracec.so", getenv("EXTRAE_HOME"));
#endif
		}
		else
		{
			if (appType->get_isOpenMP())
				sprintf (buffer, "%s/lib/lib_dyn_omptrace.so", getenv("EXTRAE_HOME"));
			else
				sprintf (buffer, "%s/lib/libseqtrace.so", getenv("EXTRAE_HOME"));
		}
		loadedModule = buffer;

		/* Load the module into the mutattee */
		cout << PACKAGE_NAME << ": Loading " << loadedModule << " into the target application" << endl;
	
		if (!file_exists (buffer))
		{
			/* If the library does not exist, terminate the mutatee and exit */
			cerr << PACKAGE_NAME << ": Cannot find the module. It must be under $EXTRAE_HOME/lib" << endl;
			appProcess->terminateExecution();
			exit (-1);
		}
		if (!appProcess->loadLibrary (loadedModule.c_str()))
		{
			/* If the library cannot be loaded, terminate the mutatee and exit */
			cerr << PACKAGE_NAME << ": Cannot load library! Retry using -v to gather more information on this error!" << endl;
			appProcess->terminateExecution();
			exit (-1);
		}

		/* Load instrumentation API patches */
		loadAPIPatches (appImage);
		if (appType->get_isMPI() && ::XML_GetTraceMPI())
			loadMPIPatches (appImage);

		InstrumentCalls (appImage, appProcess, appType, ParallelFunctions, UserFunctions, ::XML_GetTraceMPI(), ::XML_GetTraceOMP(), true);

		if (appType->get_isOpenMP())
		{
			cout << PACKAGE_NAME << ": Instrumenting OpenMP runtime" << endl;
			InstrumentOMPruntime (::XML_GetTraceOMP_locks(), appType, appImage, appProcess);
		}

		/* If the application is NOT MPI, instrument the MAIN symbol in order to
 		   initialize and finalize the instrumentation */
		if (!appType->get_isMPI())
		{
			/* Typical main entry & exit */
			wrapRoutine (appImage, appProcess, "main", "Extrae_init", "Extrae_fini");

			/* Special cases (e.g., fortran stop call) */
			string exit_calls[] =
			  {
				  "_xlfExit",
				  "_gfortran_stop_numeric",
				  "for_stop_core",
				  ""
				};

			int i = 0;
			while (exit_calls[i].length() > 0)
			{
				BPatch_function *special_exit = getRoutine (exit_calls[i], appImage);
				if (NULL != special_exit)
					wrapRoutine (appImage, appProcess, exit_calls[i], "Extrae_fini", "");
				i++;
			}
		}

		GenerateSymFile (ParallelFunctions, UserFunctions, appImage, appProcess);
	}

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

	int retVal;
	if (appProcess->terminationStatus() == ExitedNormally)
		retVal = appProcess->getExitCode();
	else if(appProcess->terminationStatus() == ExitedViaSignal)
		retVal = appProcess->getExitSignal();

	delete appProcess;

	return 0;
}
