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

#ifndef APPLICATIONTYPE_H
#define APPLICATIONTYPE_H

#include <string>

#include <BPatch.h>

using namespace std;

class ApplicationType
{
	public:
	enum MPI_type_t {MPI_C, MPI_Fortran_0u, MPI_Fortran_1u, MPI_Fortran_2u, MPI_Fortran_ucase};
	enum OMP_rte_t  {Unknown, Intel_v81, Intel_v91, IBM_v16, GNU_v42};

	void detectApplicationType (BPatch_image *appImage);
	void dumpApplicationType (void);
	bool isMangledOpenMProutine (string name);
	string demangleOpenMProutine (string name);

	inline bool get_isMPI (void) const;
	inline bool get_isOpenMP (void) const;
	inline OMP_rte_t get_OpenMP_rte (void) const;

	private:
	bool isMPI;
	bool isOpenMP;
	MPI_type_t MPI_type;
	OMP_rte_t OpenMP_runtime;

	string TranslatePFToUF (string PF, OMP_rte_t type);
	OMP_rte_t checkIntelOpenMPRuntime (BPatch_image *appImage);
};

inline bool ApplicationType::get_isMPI (void) const
{
	return isMPI;
}

inline bool ApplicationType::get_isOpenMP (void) const
{
	return isOpenMP;
}

inline ApplicationType::OMP_rte_t ApplicationType::get_OpenMP_rte (void) const
{
	return OpenMP_runtime;
}

#endif /* APPLICATIONTYPE_H */

