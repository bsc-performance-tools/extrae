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

#define SPLIT_CHAR '+'
void discoverInstrumentationLevel(set<string> & UserFunctions, map<string, vector<string> > & LoopLevels);

static BPatch *bpatch;

static char *excludeUF = NULL; /* Fitxer que indica quines User Functions s'exclouen */
static char *includeUF = NULL; /* Fitxer que indica quines User Functions s'inclouen */

static char *configXML = NULL; /* XML configuration file */

static int VerboseLevel = 0;  /* Verbose Level */
static bool useHWC = false;
static bool ListFunctions = false;
static bool extrae_detecting_application_type = false;
static bool BinaryLinkedWithInstrumentation = false;
static bool BinaryRewrite = false;
static bool DecodeBasicBlock = false;
static string loadedModule;

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

void errorFunc(BPatchErrorLevel level, int num, const char* const* params)
{
	/* Do not print certain "detection" messages if verbose level is 1/2 only */
	if ((VerboseLevel == 1 && !extrae_detecting_application_type) || VerboseLevel == 2)
	{
		if (num == 0)
		{
			// conditional reporting of warnings and informational messages
			if (level != BPatchInfo)
				cerr << PACKAGE_NAME": " << params[0] << endl;
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
					cerr << PACKAGE_NAME": Error #" << num << " (level " << level << "): " << line << endl;
			}
		}
	}
}

void ExecCallback (BPatch_thread *thread)
{
	cout << "Process " << thread->getProcess()->getPid() << " is executing exec() call" << endl;
}

static void GenerateSymFile (set<string> &ParFunc, set<string> &UserFunc, BPatch_image *appImage, BPatch_addressSpace *appProces)
{
	ofstream symfile;
	string symname = string(::XML_GetFinalDirectory())+string("/")+string(::XML_GetTracePrefix())+".sym";

	symfile.open (symname.c_str());
	if (!symfile.good())
	{
		cerr << "Cannot create the symbolic file" << symname << endl;
		return;
	}

	for (set<string>::iterator iter = ParFunc.begin();
		iter != ParFunc.end(); iter++)
	{
		BPatch_function *f = getRoutine ((*iter).c_str(), appImage);

		if (f != NULL)
		{
			BPatch_Vector< BPatch_statement > lines;

			appProces->getSourceLines ((unsigned long) f->getBaseAddr(), lines);
			if (lines.size() > 0)
			{
				symfile << "P " << hex << f->getBaseAddr() << dec << " \"" << *iter
					<< "\" \"" <<  lines[0].fileName() <<  "\" " << lines[0].lineNumber()
					<< endl;
			}
			else
			{
				/* this happens if the application was not compiled with -g */
				char modname[1024];
				f->getModuleName (modname, 1024);
				symfile << "P " << hex << f->getBaseAddr() << dec << " \"" << *iter
					<< "\" \"" << modname << "\" 0" << endl;
			}
		}
	}

	for (set<string>::iterator iter = UserFunc.begin();
		iter != UserFunc.end(); iter++)
	{
		BPatch_function *f = getRoutine ((*iter).c_str(), appImage);

		if (f != NULL)
		{
			BPatch_Vector< BPatch_statement > lines;

			appProces->getSourceLines ((unsigned long) f->getBaseAddr(), lines);
			if (lines.size() > 0)
			{
				symfile << "U " << hex << f->getBaseAddr() << dec << " \"" << *iter
					<< "\" \"" << lines[0].fileName() <<  "\" " << lines[0].lineNumber()
					<< endl;
			}
			else
			{
				/* this happens if the application was not compiled with -g */
				char modname[1024];
				f->getModuleName (modname, 1024);
				symfile << "U " << hex << f->getBaseAddr() << dec << " \"" << *iter
					<< "\" \"" << modname << "\" 0" << endl;
			}
		}
	}

    map<string, unsigned>::iterator BB_symbols_iter = BB_symbols->begin();
    map<string, unsigned>::iterator BB_symbols_end = BB_symbols->end();
    while(BB_symbols_iter != BB_symbols_end){
        symfile << "b " << BB_symbols_iter->second << " \"" << BB_symbols_iter->first << "\"\n";
        BB_symbols_iter++;
    }



	symfile.close();
}

char * printUsage()
{
    char * buffer = (char*) malloc(1024*sizeof(char*));
    strcat(buffer, "Extrae dyninst utility\n\n");
    strcat(buffer, "Usage: extrae OPTIONS binary\n\n");
    strcat(buffer, "OPTIONS:\n");
    strcat(buffer, "-exclude              Exclude functions from the instrumentation.\n");
    strcat(buffer, "-include              Include functions to be instrumented.\n");
#if defined(ALLOW_EXCLUDE_PARALLEL)
    strcat(buffer, "-exclude-parallel     Exclude parallel\n");
#endif
    strcat(buffer, "-config               Path to Extrae XML config file.\n");
    strcat(buffer, "-counters             Instrument hardware counters.\n");
    strcat(buffer, "-list-functions       List functions found in the binary/process image.\n");
    strcat(buffer, "-rewrite              Rewrite the binay to link with the extrae instrumentation.\n");
    strcat(buffer, "-decodeBB             Decode the BB for the selected user functions identifying loops.\n");
    strcat(buffer, "-v                    Verbose\n");

    return buffer;
}


static int processParams (int argc, char *argv[])
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
        if ((strcmp (argv[i], "-h") == 0) || (strcmp (argv[i], "--help") == 0))
        {
            printf("%s\n", printUsage());
            exit(-1);
        }
		else if (strcmp (argv[i], "-exclude") == 0)
		{
			i++;
			excludeUF = argv[i];
			i++;
			leave = (i >= argc);
		}
		else if (strcmp (argv[i], "-include") == 0)
		{
			i++;
			includeUF = argv[i];
			i++;
			leave = (i >= argc);
		}
#if defined(ALLOW_EXCLUDE_PARALLEL)
		else if (strcmp (argv[i], "-exclude-parallel") == 0)
		{
			i++;
			excludePF = argv[i];
			i++;
			leave = (i >= argc);
		}
#endif
		else if (strcmp (argv[i], "-config") == 0)
		{
			i++;
			configXML = argv[i];
			i++;
			leave = (i >= argc);
		}
		else if (strcmp(argv[i], "-counters") == 0)
		{
			useHWC = true;
			i++;
			leave = (i >= argc);
		}
		else if (strcmp (argv[i], "-v") == 0)
		{
			VerboseLevel++;
			i++;
			leave = (i >= argc);
		}
		else if (strcmp (argv[i], "-list-functions") == 0)
		{
			ListFunctions = true;
			i++;
			leave = (i >= argc);
		}
		else if (strcmp (argv[i], "-rewrite") == 0)
		{
			BinaryRewrite = true;
			i++;
			leave = (i >= argc);
		}
        else if (strcmp (argv[i], "-decodeBB") == 0)
        {
            DecodeBasicBlock = true;
            i++;
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

static void ReadFileIntoList (char *fitxer, set<string>& container)
{
	char str[2048];

	fstream file_op (fitxer, ios::in);
	if (!file_op.good())
	{
		cerr << PACKAGE_NAME << ": Error! Cannot open file " << fitxer << endl;
		return;
	}

	if (VerboseLevel >= 2)
		cout << PACKAGE_NAME << ": Read";

	while (file_op >> str)
	{
		container.insert (str);
		if (VerboseLevel >= 2)
			cout << " " << str;
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
			     << "    In shared library? " << (f->isSharedLib()?"yes":"no") << endl
                 << "    Number of BB: " << getBasicBlocksSize(f) << endl; 

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

static void InstrumentCalls (BPatch_image *appImage, BPatch_addressSpace *appProcess,
	ApplicationType *appType, set<string> &OMPset, set<string> &USERset,
    map<string, vector<string> > & LoopLevels,
    bool instrumentMPI, bool instrumentOMP, bool instrumentUF)
{
	unsigned i = 0;
	unsigned OMPinsertion = 0;
	unsigned OMPreplacement_intel_v11 = 0;
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

	set<string> CUDAkernels;

	/* Look for CUDA kernels if the application is CUDA */
	if (appType->get_isCUDA())
	{
		cout << PACKAGE_NAME << ": Looking for CUDA kernels inside binary (this may take a while)..." << endl;

		i = 0;
		while (i < vfunctions->size())
		{
			char name[1024];

			BPatch_function *f = (*vfunctions)[i];
			f->getName (name, sizeof(name));
			BPatch_Vector<BPatch_point *> *vpoints = f->findPoint (BPatch_subroutine);

			if (vpoints != NULL)
			{
				unsigned j = 0;
				while (j < vpoints->size())
				{
					BPatch_function *called = ((*vpoints)[j])->getCalledFunction();
					if (NULL != called)
					{
						char calledname[1024];
						called->getName (calledname, 1024);
	
						if (strncmp (calledname, "__device_stub__", strlen("__device_stub__")) == 0)
						{
							CUDAkernels.insert (name);
							if (VerboseLevel)
								cout << PACKAGE_NAME << ": Found kernel " << name << endl;
						}
					}
					j++;
				}
			}
			i++;
		}
		cout << PACKAGE_NAME << ": Finished looking for CUDA kernels" << endl;
	}

	cout << PACKAGE_NAME << ": Parsing executable looking for instrumentation points (" << vfunctions->size() << ") ";
	if (VerboseLevel)
		cout << endl;
	else
		cout << flush;

	/*
	  The 1st step includes:
	  a) gather information of openmp outlined routines (original is added to USERset),
	  b) instrument openmp outlined routines
	  c) instrument mpi calls
	  d) instrument api calls
	*/

	i = 0;
	while (i < vfunctions->size())
	{
		char name[1024], sharedlibname_c[1024];

		BPatch_function *f = (*vfunctions)[i];
		(f->getModule())->getFullName (sharedlibname_c, sizeof(sharedlibname_c));
		f->getName (name, sizeof(name));

		string sharedlibname = sharedlibname_c;

		string sharedlibname_ext;
		if (sharedlibname.rfind('.') != string::npos)
			sharedlibname_ext = sharedlibname.substr (sharedlibname.rfind('.'));
		else
			sharedlibname_ext = "";

		/* For OpenMP apps, if the application has been linked with Extrae, just need to
		   instrument the function calls that have #pragma omp in them. The outlined
			 routines will be instrumented by the library attached to the binary */
		if (!BinaryLinkedWithInstrumentation &&
		    instrumentOMP && appType->get_isOpenMP() && loadedModule != sharedlibname)
		{
			/* OpenMP instrumentation (just for OpenMP apps) */
			if (appType->isMangledOpenMProutine (name))
			{
				if (VerboseLevel)
					if (!BinaryLinkedWithInstrumentation)
						cout << PACKAGE_NAME << ": Instrumenting OpenMP outlined routine " << name << endl;

				if (!BinaryLinkedWithInstrumentation)
				{
					/* Instrument routine */ 
					wrapTypeRoutine (f, name, OMPFUNC_EV, appImage);

					/* Add to list if not already there */
					OMPset.insert (name);
				}

				/* Demangle name and add into the UF list if it didn't exist there */
				string demangled = appType->demangleOpenMProutine (name);
				if (!XML_excludeAutomaticFunctions())
					USERset.insert (demangled);
				if (VerboseLevel)
				{
					if (!XML_excludeAutomaticFunctions())
						cout << PACKAGE_NAME << ": Adding demangled OpenMP routine " << demangled << " to the user function list" << endl;	
					else
						cout << PACKAGE_NAME << ": Will not add demangled OpenMP routine " << demangled << " due to user request in the XML configuration file" << endl;
				}

				OMPinsertion++;
			}
		}

		if (sharedlibname_ext == ".f" || sharedlibname_ext == ".F" ||   /* fortran */
		  sharedlibname_ext == ".for" || sharedlibname_ext == ".FOR" || /* fortran */
		  sharedlibname_ext == ".f90" || sharedlibname_ext == ".F90" || /* fortran 90 */
		  sharedlibname_ext == ".i90" ||                                /* fortran 90 through ifort */
		  sharedlibname_ext == ".f77" || sharedlibname_ext == ".F77" || /* fortran 77 */
		  sharedlibname_ext == ".c" || sharedlibname_ext == ".C" ||     /* C */
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
								cout << PACKAGE_NAME << ": Replaced call " << calledname << " in routine " << name << " (" << sharedlibname << ")" << endl;
						}
						else
							cerr << PACKAGE_NAME << ": Cannot replace " << calledname << " routine" << endl;
					}

					/* Check MPI calls */
					if (!BinaryLinkedWithInstrumentation &&
					    instrumentMPI && appType->get_isMPI() && (
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

					/* Special instrumentation for calls in Intel OpenMP runtime v11/v12
					   currently only for __kmpc_fork_call */
					if (!BinaryLinkedWithInstrumentation &&
					    appType->get_OpenMP_rte() == ApplicationType::Intel_v11 &&
					    strncmp (calledname, "__kmpc_fork_call", strlen("__kmpc_fork_call")) == 0)
					{
						BPatch_function *patch_openmp = getRoutine (
							"__kmpc_fork_call_extrae_dyninst", appImage, false);
						if (patch_openmp != NULL)
						{
							if (appProcess->replaceFunctionCall (*((*vpoints)[j]), *patch_openmp))
							{
								OMPreplacement_intel_v11++;
								if (VerboseLevel)
									cout << PACKAGE_NAME << ": Replaced call " << calledname << " in routine " << name << " (" << sharedlibname << ")" << endl;
							}
							else
								cerr << PACKAGE_NAME << ": Cannot replace " << calledname << " routine" << endl;
						}

						/* Instrument the routine that invokes the runtime */
						if (!XML_excludeAutomaticFunctions())
							USERset.insert (name);
						if (VerboseLevel)
						{
							if (!XML_excludeAutomaticFunctions())
								cout << PACKAGE_NAME << ": Adding call to OpenMP routine " << name << " to the user function list" << endl;
							else
								cout << PACKAGE_NAME << ": Will not add call to OpenMP routine " << name << " due to user request in the XML configuration file" << endl;
						}
					}

					/* Special instrumentation for fork() / wait() / exec* calls */
					if (!BinaryLinkedWithInstrumentation &&
					   (
					    strncmp (calledname, "fork", strlen("fork")) == 0 ||
					    strncmp (calledname, "wait", strlen("wait")) == 0 ||
					    strncmp (calledname, "waitpid", strlen("waitpid")) == 0 ||
					    strncmp (calledname, "system", strlen("system")) == 0 ||
					    strncmp (calledname, "execl", strlen("execl")) == 0 ||
					    strncmp (calledname, "execle", strlen("execle")) == 0 ||
					    strncmp (calledname, "execlp", strlen("execlp")) == 0 ||
					    strncmp (calledname, "execv", strlen("execv")) == 0 ||
					    strncmp (calledname, "execve", strlen("execve")) == 0 ||
					    strncmp (calledname, "execvp", strlen("execvp")) == 0
						)
					   )
					{
						/* Instrument the routine that invokes the runtime */
						if (!XML_excludeAutomaticFunctions())
							USERset.insert (name);
						if (VerboseLevel)
						{
							if (!XML_excludeAutomaticFunctions())
								cout << PACKAGE_NAME << ": Adding routine " << name << " to the user function list because it calls to " << calledname << endl;
							else
								cout << PACKAGE_NAME << ": Will not add routine to the user function list " << name << " due to user request in the XML configuration file" << endl;
						}
					}

					/* Instrument routines that call CUDA */
					if (appType->get_isCUDA())
					{
						string scalledname (calledname);

						if (find (CUDAkernels.begin(), CUDAkernels.end(), scalledname) != CUDAkernels.end())
						{
							if (!XML_excludeAutomaticFunctions())
								USERset.insert (name);

							if (VerboseLevel)
							{
								if (!XML_excludeAutomaticFunctions())
									cout << PACKAGE_NAME << ": Adding routine " << name << " to the user function list because it calls the CUDA kernel '" << calledname<< "'" << endl;	
								else
									cout << PACKAGE_NAME << ": Will not instrument CUDA routine " << name << " due to user request in the XML configuration file" << endl;
							}
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

	if (USERset.size() > 0 && instrumentUF)
	{
		/* Instrument user functions! */

		cout << PACKAGE_NAME << ": Instrumenting user functions...";
		if (VerboseLevel)
			cout << endl;
		else
			cout << flush;

		set<string>::iterator iter = USERset.begin();
		while (iter != USERset.end())
		{
			if (*iter != "main")
			{
				BPatch_function *f = getRoutine ((*iter).c_str(), appImage);

				if (f != NULL)
				{
					wrapTypeRoutine (f, *iter, USRFUNC_EV, appImage);

                    vector<string> points = LoopLevels[*iter]; // LoopLevels['foo'] = [bb_1,loop_1.2.3,bb_5]
                    instrumentLoops(f, *iter, appImage, points);
                    instrumentBasicBlocks(f, appImage, points);

					UFinsertion++;

					if (VerboseLevel)
						cout << PACKAGE_NAME << ": Instrumenting user function : " << *iter << endl;
				}
				else
				{
					if (VerboseLevel)
						cout << PACKAGE_NAME << ": Unable to instrument user function : " << *iter << endl;
				}
			}
			else
			{
				if (VerboseLevel)
					cout << PACKAGE_NAME << ": Although 'main' symbol was in the instrumented functions list, it will not be instrumented" << endl;
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
	{
		cout << PACKAGE_NAME << ": " << OMPinsertion << " OpenMP patches applied to outlined routines" << endl;
		if (appType->get_OpenMP_rte() == ApplicationType::Intel_v11)
			cout << PACKAGE_NAME << ": " << OMPreplacement_intel_v11 << " OpenMP patches applied to specific locations for Intel runtime" << endl;
	}
	if (USERset.size() > 0)
		cout << PACKAGE_NAME << ": " << UFinsertion << " user function" << ((UFinsertion!=1)?"s":"") << " instrumented" << endl;
}

int main (int argc, char *argv[])
{
	set<string> UserFunctions;
	set<string> ParallelFunctions;
    map<string, vector<string> > LoopLevels;
	char *env_var;

	int index;

	if (getenv("EXTRAE_HOME") == NULL)
	{
		cerr << PACKAGE_NAME << ": Environment variable EXTRAE_HOME is undefined" << endl;
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

	/* Parse the params */
	index = processParams (argc, argv);

	char * envvar_dyn = (char *) malloc ((strlen("EXTRAE_DYNINST_RUN=yes")+1)*sizeof (char));
	if (NULL == envvar_dyn)
	{
		cerr << PACKAGE_NAME << ": Error! Unable to allocate memory for EXTRAE_DYNINST_RUN environment variable" << endl;
		exit (-1);
	}
	sprintf (envvar_dyn, "EXTRAE_DYNINST_RUN=yes");
	putenv (envvar_dyn);

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

	cout << "Welcome to " << PACKAGE_STRING  << " revision " << EXTRAE_SVN_REVISION
	  << " based on " << EXTRAE_SVN_BRANCH << " launcher using DynInst "
	  << DYNINST_MAJOR << "." << DYNINST_MINOR << "." << DYNINST_SUBMINOR << endl;

	int i = 1;
	while (argv[index+i] != NULL)
	{
		cout << PACKAGE_NAME << ": Argument " << i <<  " - " << argv[index+i] << endl;
		i++;
	}

	BPatch_process *appProcess = NULL;
	BPatch_binaryEdit *appBin = NULL;
	BPatch_addressSpace *appAddrSpace = NULL;

	if (!BinaryRewrite)
	{
		cout << PACKAGE_NAME << ": Creating process for image binary " << argv[index];
		cout.flush ();
		appProcess = bpatch->processCreate ((const char*) argv[index], (const char**) &argv[index], (const char**) environ);
		if (appProcess == NULL)
		{
			cerr << endl << PACKAGE_NAME << ": Error creating the target application process" << endl;
			exit (-1);
		}
		cout << " PID(" << appProcess->getPid() << ")" << endl;

		/* Stop the execution in order to load the instrumentation library */
		cout << PACKAGE_NAME << ": Stopping mutatee execution" << endl;
		if (!appProcess->stopExecution())
		{
			cerr << PACKAGE_NAME << ": Cannot stop execution of the target application" << endl;
			exit (-1);
		}
		appAddrSpace = appProcess;
	}
	else
	{
		cout << PACKAGE_NAME << ": Rewriting binary " << argv[index] << endl;
		appBin = bpatch->openBinary ((const char*) argv[index], false); /* passed true to instrument libraries before !? */
		if (appBin == NULL)
		{
			cerr << PACKAGE_NAME << ": Error opening binary for rewriting" << endl;
			exit (-1);
		}
		appAddrSpace = appBin;
	}

	BPatch_image *appImage = appAddrSpace->getImage();
	if (appImage == NULL)
	{
		cerr << PACKAGE_NAME << ": Error while acquiring application image" << endl;
		exit (-1);
	}

	/* The user asks for the list of functions, simply show it */
	if (ListFunctions)
	{
		ShowFunctions (appImage);
		if (!BinaryRewrite)		
			appProcess->terminateExecution();
		exit (-1);
	}

	/* Read files */
	if (::XML_have_UFlist())
	{
		if (VerboseLevel)
			cout << PACKAGE_NAME << ": Reading instrumented user functions from " << ::XML_UFlist() << endl;
		ReadFileIntoList (::XML_UFlist(), UserFunctions);
        discoverInstrumentationLevel(UserFunctions, LoopLevels);
	}

    if (DecodeBasicBlock)
    {
        std::set<string>::iterator f_begin = UserFunctions.begin();
        std::set<string>::iterator f_end = UserFunctions.end();
        while (f_begin != f_end)
        {
            BPatch_function *f = getRoutine ((*f_begin).c_str(), appImage);
            if (f == NULL)
            {
                cerr << PACKAGE_NAME << ": Unable to find " << *f_begin << " function!" << endl;
            }
            else
            {
                cout<<decodeBasicBlocks(f, *f_begin);
            }
            f_begin++;
        }
		if (!BinaryRewrite)	{	
			appProcess->terminateExecution();
            }
        exit(-1);
    }

	if (::XML_CheckTraceEnabled())
	{
		ApplicationType *appType = new ApplicationType ();

		extrae_detecting_application_type = true;

		cout << PACKAGE_NAME << ": Detecting application type " << endl;
		appType->detectApplicationType (appImage);
		appType->dumpApplicationType ();

		cout << PACKAGE_NAME << ": Detecting whether the application has been already linked with Extrae : ";
		BPatch_function *extrae_init = getRoutine ("Extrae_init", appImage, false);
		BinaryLinkedWithInstrumentation = extrae_init != NULL;
		cout << (BinaryLinkedWithInstrumentation?"yes":"no") << endl;

		extrae_detecting_application_type = false;

		/* If the application has not been linked with instrumentation library, load the
		   appropriate module */
		if (!BinaryLinkedWithInstrumentation)
		{
			char buffer[1024]; /* will hold the library to load */

			/* Check for the correct library to be loaded */
			if (appType->get_isMPI())
			{
				if (appType->get_isOpenMP())
				{
					if (appType->get_MPI_type() == ApplicationType::MPI_C)
						sprintf (buffer, "%s/lib/lib_dyn_ompitracec-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
					else
						sprintf (buffer, "%s/lib/lib_dyn_ompitracef-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
				}
				else if (appType->get_isCUDA())
				{
					if (appType->get_MPI_type() == ApplicationType::MPI_C)
						sprintf (buffer, "%s/lib/lib_dyn_cudampitracec-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
					else
						sprintf (buffer, "%s/lib/lib_dyn_cudampitracef-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
				}
				else
				{
					if (appType->get_MPI_type() == ApplicationType::MPI_C)
						sprintf (buffer, "%s/lib/lib_dyn_mpitracec-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
					else
						sprintf (buffer, "%s/lib/lib_dyn_mpitracef-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
				}
			}
			else
			{
				if (appType->get_isOpenMP())
				{
					sprintf (buffer, "%s/lib/libomptrace-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
				}
				else
				{
					if (appType->get_isCUDA())
						sprintf (buffer, "%s/lib/libcudatrace-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
					else
						sprintf (buffer, "%s/lib/libseqtrace-%s.so", getenv("EXTRAE_HOME"), PACKAGE_VERSION);
				}
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
			if (!appAddrSpace->loadLibrary (loadedModule.c_str()))
			{
				/* If the library cannot be loaded, terminate the mutatee and exit */
				cerr << PACKAGE_NAME << ": Cannot load library! Retry using -v to gather more information on this error!" << endl;
				appProcess->terminateExecution();
				exit (-1);
			}
		} /* ! BinaryLinkedWithInstrumentation */
		else
			cout << PACKAGE_NAME << ": The application seems to be linked with Extrae libraries. Won't load additional libraries..." << endl;

		/* Load instrumentation API patches */
		loadAPIPatches (appImage);
		if (appType->get_isMPI() && ::XML_GetTraceMPI())
			loadMPIPatches (appImage);

		/* Instrument fork, wait, waitpid and exec calls */
		InstrumentForks (appImage);

		/* Apply instrumentation of runtimes only if not linked with Extrae */
		if (!BinaryLinkedWithInstrumentation && appType->get_isOpenMP())
		{
			if (appType->get_OpenMP_rte() == ApplicationType::Intel_v11)
			{
				cout << PACKAGE_NAME << ": Gathering information for Intel v11 OpenMP runtime" << endl;
# warning "Aixo nomes es per !BinaryRewriting!"
				InstrumentOMPruntime_Intel (appImage, appProcess);
			}
			cout << PACKAGE_NAME << ": Instrumenting OpenMP runtime" << endl;
			InstrumentOMPruntime (::XML_GetTraceOMP_locks(), appType, appImage);
		}

		/* Apply instrumentation of runtimes only if not linked with Extrae */
		if (!BinaryLinkedWithInstrumentation && appType->get_isCUDA())
		{
			cout << PACKAGE_NAME << ": Instrumenting CUDA runtime" << endl;
			InstrumentCUDAruntime (appType, appImage);
		}

		/* If the application is NOT MPI, instrument the MAIN symbol in order to
 		   initialize and finalize the instrumentation */
		/* Apply instrumentation of runtimes only if not linked with Extrae */
		if (!appType->get_isMPI())
		{
			/* Typical main entry & exit */
			wrapRoutine (appImage, "main", "Extrae_init", "Extrae_fini");
		}
		else
		{
			/* Cover those cases that MPI apps do not call MPI_Finalize */
			wrapRoutine (appImage, "main", "", "Extrae_fini_last_chance_Wrapper");
		}

		{
			/* Special cases (e.g., fortran stop call) */
			string exit_calls[] =
			{
				  "exit", /* C */
				  "_xlfExit", /* Fortran IBM XL */
				  "_gfortran_stop_numeric", /* Fortran GNU */
				  "for_stop_core", /* Fortran Intel */
				  ""
			};

			/* bypass error messages given if these routines are not found */
			int i = 0;
			extrae_detecting_application_type = true;
			while (exit_calls[i].length() > 0)
			{
				BPatch_function *special_exit = getRoutine (exit_calls[i].c_str(), appImage, false);
				if (NULL != special_exit)
					wrapRoutine (appImage, exit_calls[i], "Extrae_fini_last_chance_Wrapper", "");
				i++;
			}
			extrae_detecting_application_type = false;
		}

		InstrumentCalls (appImage, appAddrSpace, appType, ParallelFunctions,
		  UserFunctions, LoopLevels, ::XML_GetTraceMPI(), ::XML_GetTraceOMP(), true);

		GenerateSymFile (ParallelFunctions, UserFunctions, appImage,
		  appAddrSpace);
	}

	// bpatch->registerExecCallback(ExecCallback);

	if (!BinaryRewrite)
	{
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
	{
		string newfile = string(argv[index])+".extrae";
		cout << PACKAGE_NAME << ": Generating the instrumented binary" << endl;
		if (appBin->writeFile (newfile.c_str()))
			cout << PACKAGE_NAME << ": Congratulations " << newfile << " has been generated" << endl;
		else
			cout << PACKAGE_NAME << ": Error! Could not generate " << newfile << endl;
		
		delete appBin;
	}

	return 0;
}

void discoverInstrumentationLevel(set<string> & UserFunctions, map<string, vector<string> > & LoopLevels)
{
    set<string>::iterator uf_it = UserFunctions.begin();
    set<string>::iterator uf_end = UserFunctions.end();
    while(uf_it != uf_end)
    {
        vector<string> tokens;
        string uf = *uf_it;
        split(tokens, uf, SPLIT_CHAR);
        if (tokens.size() > 1)
        {
            string uf_name = tokens[0]; // [kernel]+loop_1.1,loop_1.2
            string bb_part = tokens[1]; // kernel+[loop_1.1,loop_1.2]
            tokens.clear();
            split(tokens, bb_part, ',');
            for(unsigned i=0; i < tokens.size(); i++)
            {
                LoopLevels[uf_name].push_back(tokens[i]);
            }
            UserFunctions.erase(uf_it);
            UserFunctions.insert(uf_name);
        }
        uf_it++;
    }

    uf_it = UserFunctions.begin();
    uf_end = UserFunctions.end();
    cout<< PACKAGE_NAME": XML functions parsed: ";
    while(uf_it != uf_end){
        if (LoopLevels.find(*uf_it) != LoopLevels.end())
        {
            cout<< "\""<<*uf_it<<"\"" <<"->[loops:";
            for(unsigned i=0; i < LoopLevels[*uf_it].size(); i++){
                cout<<LoopLevels[*uf_it][i]<<",";
            }
            cout<<"]";
        }
        else
        {
            cout<< "\""<<*uf_it<<"\"";
        }
        
        set<string>::iterator uf_it_plus_one = uf_it;
        uf_it_plus_one++;
        if (uf_it_plus_one != uf_end)
        {
            cout<<";";
        } else cout<<endl;
        uf_it++;
    }
}


