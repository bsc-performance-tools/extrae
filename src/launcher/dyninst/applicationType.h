/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/launcher/dyninst/applicationType.h,v $
 | 
 | @last_commit: $Date: 2009/01/07 14:40:25 $
 | @version:     $Revision: 1.4 $
 | 
 | History:
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

	inline const bool get_isMPI (void) const;
	inline const bool get_isOpenMP (void) const;
	inline const OMP_rte_t get_OpenMP_rte (void) const;

	private:
	bool isMPI;
	bool isOpenMP;
	MPI_type_t MPI_type;
	OMP_rte_t OpenMP_runtime;

	string TranslatePFToUF (string PF, OMP_rte_t type);
	OMP_rte_t checkIntelOpenMPRuntime (BPatch_image *appImage);
};

inline const bool ApplicationType::get_isMPI (void) const
{
	return isMPI;
}

inline const bool ApplicationType::get_isOpenMP (void) const
{
	return isOpenMP;
}

inline const ApplicationType::OMP_rte_t ApplicationType::get_OpenMP_rte (void) const
{
	return OpenMP_runtime;
}

#endif /* APPLICATIONTYPE_H */

