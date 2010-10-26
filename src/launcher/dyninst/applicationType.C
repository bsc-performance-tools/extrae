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

#include "applicationType.h"

#include <BPatch.h>
#include <BPatch_function.h>

#include <iostream>

#define INTEL_PARLOOP_RTNS     ".*_[0-9]+__par_loop[0-9]+.*"
#define INTEL_PARRGN_RTNS      ".*_[0-9]+__par_region[0-9]+.*"

string ApplicationType::TranslatePFToUF (string PF, OMP_rte_t type)
{
	string result;

	if (type == Intel_v81 || type == Intel_v91)
	{
		string tmp1 = PF.substr (0, PF.rfind("__par_"));
		string tmp2 = tmp1.substr (0, tmp1.rfind("_"));

		if (type == Intel_v81)
			result = tmp2.substr (1, tmp2.length()-1);
		else 
			result = tmp2.substr (2, tmp2.length()-1);
	}
	else if (type == IBM_v16)
	{
		result = PF.substr (0, PF.find ("$.OL$."));
	}
	else if (type == GNU_v42)
	{
		result = PF.substr (0, PF.find (".omp_fn."));
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
		string routinev81 = TranslatePFToUF (functionName, Intel_v81);
		string routinev91 = TranslatePFToUF (functionName, Intel_v91);

		if (appImage->findFunction (routinev81.c_str(), found_funcs) != NULL)
			if (found_funcs.size() > 0)
				return Intel_v81;

		if (appImage->findFunction (routinev91.c_str(), found_funcs) != NULL)
			if (found_funcs.size() > 0)
				return Intel_v91;
	}

	return Unknown;
}

void ApplicationType::detectApplicationType (BPatch_image *appImage)
{
	BPatch_Vector<BPatch_function *> found_funcs;

	isOpenMP = isMPI = false;


	/* Check for different implementations of OpenMP rte */
	if ((appImage->findFunction ("__kmpc_fork_call", found_funcs) != NULL) ||
	    (appImage->findFunction ("__kmpc_invoke_task_func", found_funcs) != NULL) )
	{
		OpenMP_runtime = checkIntelOpenMPRuntime (appImage);
		isOpenMP = true;
	}
	else if ((appImage->findFunction ("_xlsmpParallelDoSetup_TPO", found_funcs) != NULL) ||
	         (appImage->findFunction ("_xlsmpParRegionSetup_TPO", found_funcs) != NULL) ||
	         (appImage->findFunction ("_xlsmpWSDoSetup_TPO", found_funcs) != NULL))
	{
		OpenMP_runtime = IBM_v16;
		isOpenMP = true;
	}
	else if ((appImage->findFunction ("GOMP_parallel_start", found_funcs) != NULL) ||
	         (appImage->findFunction ("GOMP_loop_end", found_funcs) != NULL) ||
	         (appImage->findFunction ("GOMP_barrier", found_funcs) != NULL))
	{
		OpenMP_runtime = GNU_v42;
		isOpenMP = true;
	}

	/* MPI applications must have MPI_Init (or mpi_init fortran names) */
	if ((appImage->findFunction ("MPI_Init", found_funcs) != NULL) ||
	    (appImage->findFunction ("mpi_init", found_funcs) != NULL) ||
	    (appImage->findFunction ("mpi_init_", found_funcs) != NULL) ||
	    (appImage->findFunction ("mpi_init__", found_funcs) != NULL) ||
	    (appImage->findFunction ("MPI_INIT", found_funcs) != NULL))
	{
		isMPI = true;
		if (appImage->findFunction ("MPI_INIT", found_funcs) != NULL)
			MPI_type = MPI_Fortran_ucase;
		else if (appImage->findFunction ("mpi_init__", found_funcs) != NULL)
			MPI_type = MPI_Fortran_2u;
		else if (appImage->findFunction ("mpi_init_", found_funcs) != NULL)
			MPI_type = MPI_Fortran_1u;
		else if (appImage->findFunction ("mpi_init", found_funcs) != NULL)
			MPI_type = MPI_Fortran_0u;
		else if (appImage->findFunction ("mpi_init__", found_funcs) != NULL)
			MPI_type = MPI_C;
	}
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
		else if (OpenMP_runtime == Intel_v81)
			cout << "(Intel 8.1) ";
		else if (OpenMP_runtime == Intel_v91)
			cout << "(Intel >= 9.1) ";
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
		else if (MPI_type == MPI_Fortran_0u)
			cout << "(Fortran language with 1 underscore) ";
		else if (MPI_type == MPI_Fortran_0u)
			cout << "(Fortran language with 2 underscores) ";
		else if (MPI_type == MPI_Fortran_ucase)
			cout << "(Fortran language in uppercase) ";
	}
	if (!isOpenMP && !isMPI)
		cout << "Sequential";
	cout << endl;
}

bool ApplicationType::isMangledOpenMProutine (string name)
{
	bool result = false;

	if (OpenMP_runtime == Intel_v81 || OpenMP_runtime == Intel_v91)
	{
		result = name.find ("__par_loop") != string::npos
		         || 
		         name.find ("__par_region") != string::npos;
	}
	else if (OpenMP_runtime == IBM_v16)
	{
		result = name.find ("$.OL$.") != string::npos;
	}
	else if (OpenMP_runtime == GNU_v42)
	{
		result = name.find (".omp_fn.") != string::npos;
	}

	return result;
}
