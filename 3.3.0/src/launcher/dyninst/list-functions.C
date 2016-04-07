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
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif
#if HAVE_LIBGEN_H
# include <libgen.h>
#endif

using namespace std; 

#include <BPatch.h>
#include <BPatch_statement.h>
#include <BPatch_function.h>
#include <BPatch_point.h>
#include <BPatch_process.h>
#include <BPatch_point.h>

#define SPLIT_CHAR '+'
void discoverInstrumentationLevel(set<string> & UserFunctions, map<string, vector<string> > & LoopLevels);

static BPatch *bpatch;

#define DYNINST_NO_ERROR -1

/******************************************************************************
 **      Function name : file_exists (char*)
 **      Author : HSG
 **      Description : Checks whether a file exists
 ******************************************************************************/
static int file_exists (char *fname)
{
#if defined(HAVE_ACCESS) || 1
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

int main (int argc, char *argv[])
{
	char *env_var;

	if (argc != 2)
	{
		cerr << PACKAGE_NAME << ": Invalid number of parameters" << endl;
		return 1;
	}

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

	char * envvar_dyn = (char *) malloc ((strlen("EXTRAE_DYNINST_RUN=yes")+1)*sizeof (char));
	if (NULL == envvar_dyn)
	{
		cerr << PACKAGE_NAME << ": Error! Unable to allocate memory for EXTRAE_DYNINST_RUN environment variable" << endl;
		exit (-1);
	}
	sprintf (envvar_dyn, "EXTRAE_DYNINST_RUN=yes");
	putenv (envvar_dyn);

	/* Does the binary exists? */
	if (!file_exists(argv[1]))
	{
		cout << PACKAGE_NAME << ": Executable " << argv[1] << " cannot be found!" << endl;
		exit (-1);
	}

	/* Create an instance of the BPatch library */
	bpatch = new BPatch;

	/* Don't check recursion in snippets */
	bpatch->setTrampRecursive (true);

	BPatch_process *appProcess = NULL;
	BPatch_addressSpace *appAddrSpace = NULL;

	cout << PACKAGE_NAME << ": Creating process for image binary " << argv[1];
	cout.flush ();
	appProcess = bpatch->processCreate ((const char*) argv[1], (const char**) &argv[1], (const char**) environ);
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

	BPatch_image *appImage = appAddrSpace->getImage();
	if (appImage == NULL)
	{
		cerr << PACKAGE_NAME << ": Error while acquiring application image" << endl;
		exit (-1);
	}

	ShowFunctions (appImage);
	appProcess->terminateExecution();

	return 0;
}
