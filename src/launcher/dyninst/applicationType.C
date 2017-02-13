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

#include "applicationType.h"

#include <BPatch.h>
#include <BPatch_function.h>

#include <commonSnippets.h>

#include <iostream>

/* For Intel 8.x - 9.x (maybe 10.x?) */
#define INTEL_PARLOOP_RTNS     ".*_[0-9]+__par_loop[0-9]+.*"
#define INTEL_PARRGN_RTNS      ".*_[0-9]+__par_region[0-9]+.*"

string ApplicationType::TranslatePFToUF (string PF, OMP_rte_t type)
{
	string result;

	if (type == Intel_v8_1 || type == Intel_v9_1)
	{
		string tmp1 = PF.substr (0, PF.rfind("__par_"));
		string tmp2 = tmp1.substr (0, tmp1.rfind("_"));

		if (type == Intel_v8_1)
			result = tmp2.substr (1, tmp2.length()-1);
		else 
			result = tmp2.substr (2, tmp2.length()-1);
	}
	else if (type == Intel_v11)
	{
		/* We don't know how to handle this, if possible */
		result = PF;
	}
	else if (type == IBM_v16)
	{
		result = PF.substr (0, PF.find ("$.OL$."));
	}
	else if (type == GNU_v42)
	{
		if (PF.find (".omp_fn.") != string::npos)
			result = PF.substr (0, PF.find (".omp_fn."));
		else if (PF.find ("._omp_fn.") != string::npos)
			result = PF.substr (0, PF.find ("._omp_fn."));
		else /* unhandled */
			result = PF;
	}

	return result;
}

string ApplicationType::demangleOpenMProutine (string name)
{
	return TranslatePFToUF (name, OpenMP_runtime);
}

ApplicationType::OMP_rte_t ApplicationType::checkIntelOpenMPRuntime (BPatch_image *appImage)
{
	BPatch_Vector<BPatch_function *> found_funcs;

	/* Check if there's any OpenMP parallel routine in the binary */
	if (appImage->findFunction (INTEL_PARLOOP_RTNS, found_funcs) != NULL)
		if (found_funcs.size() == 0)
			appImage->findFunction (INTEL_PARRGN_RTNS, found_funcs);

	if (found_funcs.size() > 0)
	{
		char functionName[1024];
		found_funcs[0]->getName (functionName, 1024);
		string routinev8_1 = TranslatePFToUF (functionName, Intel_v8_1);
		string routinev9_1 = TranslatePFToUF (functionName, Intel_v9_1);

		if (appImage->findFunction (routinev8_1.c_str(), found_funcs) != NULL)
			if (found_funcs.size() > 0)
				return Intel_v8_1;

		if (appImage->findFunction (routinev9_1.c_str(), found_funcs) != NULL)
			if (found_funcs.size() > 0)
				return Intel_v9_1;
	}

	/* If we don't know about the runtime, try with icc v11 */
	return Intel_v11;
}

bool ApplicationType::detectApplicationType_checkOpenMPrte (
	vector<string> &routines, string library, BPatch_image *appImage)
{
	bool result = false;

	unsigned numroutines = routines.size();
	for (unsigned r = 0; r < numroutines && !result; r++)
	{
		BPatch_Vector<BPatch_function *> found_funcs;
		found_funcs = getRoutines (routines[r], appImage);
		unsigned numfuncs = found_funcs.size();
		for (unsigned u = 0; u < numfuncs && !result; u++)
		{
			char buffer[1024];
			char *module = found_funcs[u]->getModuleName (buffer, sizeof(buffer));
			string smodule (module);
			result = smodule.find (library) == 0;
		}
	}

	return result;
}

bool ApplicationType::detectApplicationType_checkGNUOpenMPrte (BPatch_image *appImage)
{
	const char *functions[] = {"GOMP_parallel_start",
		"GOMP_loop_end",
		"GOMP_barrier"};
	vector<string> v (functions, functions+3);

	return detectApplicationType_checkOpenMPrte (v, "libgomp.so.1", appImage);
}

bool ApplicationType::detectApplicationType_checkIntelOpenMPrte (BPatch_image *appImage)
{
	const char *functions[] = {"__kmpc_fork_call",
		"__kmpc_invoke_task_func",
		"__kmpc_barrier"};
	vector<string> v (functions, functions+3);

	return detectApplicationType_checkOpenMPrte (v, "libiomp5.so", appImage);
}

bool ApplicationType::detectApplicationType_checkIBMOpenMPrte (BPatch_image *appImage)
{
	const char *functions[] = {"_xlsmpParallelDoSetup_TPO",
		"_xlsmpParRegionSetup_TPO",
		"_xlsmpWSDoSetup_TPO" };
	vector<string> v (functions, functions+3);

	return detectApplicationType_checkOpenMPrte (v, "libxlsmp.so.1", appImage);
}

void ApplicationType::detectApplicationType (BPatch_image *appImage)
{
	BPatch_Vector<BPatch_function *> found_funcs;

	isOpenMP = isMPI = isCUDA = false;

	/* Check for different implementation of OpenMP rte */
	if (detectApplicationType_checkIntelOpenMPrte (appImage))
	{
		isOpenMP = true;
		OpenMP_runtime = checkIntelOpenMPRuntime (appImage);
	}
	else if (detectApplicationType_checkGNUOpenMPrte (appImage))
	{
		isOpenMP = true;
		OpenMP_runtime = GNU_v42;
	}
	else if (detectApplicationType_checkIBMOpenMPrte (appImage))
	{
		isOpenMP = true;
		OpenMP_runtime = IBM_v16;
	}

	/* MPI applications must have MPI_Init (or mpi_init fortran names) */
	if ((appImage->findFunction ("MPI_Init", found_funcs) != NULL) ||
	    (appImage->findFunction ("mpi_init", found_funcs) != NULL) ||
	    (appImage->findFunction ("mpi_init_", found_funcs) != NULL) ||
	    (appImage->findFunction ("mpi_init__", found_funcs) != NULL) ||
	    (appImage->findFunction ("MPI_INIT", found_funcs) != NULL))
	{
		isMPI = true;
		if (appImage->findFunction ("mpi_init", found_funcs) != NULL)
			MPI_type = MPI_Fortran_0u;
		else if (appImage->findFunction ("mpi_init__", found_funcs) != NULL)
			MPI_type = MPI_Fortran_2u;
		else if (appImage->findFunction ("mpi_init_", found_funcs) != NULL)
			MPI_type = MPI_Fortran_1u;
		else if (appImage->findFunction ("MPI_INIT", found_funcs) != NULL)
			MPI_type = MPI_Fortran_ucase;
		else 
			MPI_type = MPI_C;
	}

	/* Check for two typical CUDA routines in CUDA apps */
	if ((appImage->findFunction ("cudaLaunch", found_funcs) != NULL) &&
	    (appImage->findFunction ("cudaConfigureCall", found_funcs) != NULL))
		isCUDA = true;
}

void ApplicationType::dumpApplicationType (void)
{
	cout << PACKAGE_NAME << ": Detected application type: ";
	if (isOpenMP)
	{
		cout << "OpenMP ";
		if (OpenMP_runtime == GNU_v42)
			cout << "(GNU >= 4.2) ";
		else if (OpenMP_runtime == IBM_v16)
			cout << "(IBM >= 1.6) ";
		else if (OpenMP_runtime == Intel_v8_1)
			cout << "(Intel 8.x) ";
		else if (OpenMP_runtime == Intel_v9_1)
			cout << "(Intel 9.x or 10.x) ";
		else if (OpenMP_runtime == Intel_v11)
			cout << "(Intel >= 11.x) ";
		else
			cout << "(Unknown) ";
	}
	if (isMPI)
	{
		cout << "MPI ";
		if (MPI_type == MPI_C)
			cout << "(C language) ";
		else if (MPI_type == MPI_Fortran_0u)
			cout << "(Fortran language with 0 underscores) ";
		else if (MPI_type == MPI_Fortran_1u)
			cout << "(Fortran language with 1 underscore) ";
		else if (MPI_type == MPI_Fortran_2u)
			cout << "(Fortran language with 2 underscores) ";
		else if (MPI_type == MPI_Fortran_ucase)
			cout << "(Fortran language in uppercase) ";
	}
	if (!isOpenMP && !isMPI)
		cout << "Sequential ";

	if (isCUDA)
		cout << "CUDA-accelerated";

	cout << endl;
}

bool ApplicationType::isMangledOpenMProutine (string name)
{
	bool result = false;

	if (OpenMP_runtime == Intel_v8_1 || OpenMP_runtime == Intel_v9_1)
	{
		result = name.find ("__par_loop") != string::npos
		         || 
		         name.find ("__par_region") != string::npos;
	}
	else if (OpenMP_runtime == Intel_v11)
	{
		/* We don't know how to handle these */
		result = false;
	}
	else if (OpenMP_runtime == IBM_v16)
	{
		result = name.find ("$.OL$.") != string::npos;
	}
	else if (OpenMP_runtime == GNU_v42)
	{
		result = name.find (".omp_fn.") != string::npos || name.find ("._omp_fn.") != string::npos;
	}

	return result;
}
