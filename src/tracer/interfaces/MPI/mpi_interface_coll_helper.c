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

#include "wrapper.h"
#include "mpi_wrapper.h"
#include "mpi_interface_coll_helper.h"

static unsigned MPI_CurrentOpGlobal = 0;
static unsigned MPI_NumOpsGlobals   = 0;

unsigned Extrae_MPI_getCurrentOpGlobal (void)
{
	return MPI_CurrentOpGlobal;
}

unsigned Extrae_MPI_getNumOpsGlobals (void)
{
	return MPI_NumOpsGlobals;
}

void Extrae_MPI_ProcessCollectiveCommunicator (MPI_Comm c)
{
	int res;

	PMPI_Comm_compare (MPI_COMM_WORLD, c, &res);

	if (res == MPI_IDENT || res == MPI_CONGRUENT)
	{
		MPI_CurrentOpGlobal = ++MPI_NumOpsGlobals;

		if (Extrae_getCheckControlFile())
			CheckControlFile();
		if (Extrae_getCheckForGlobalOpsTracingIntervals())
			CheckGlobalOpsTracingIntervals();
	}
	else
		MPI_CurrentOpGlobal = 0;
}

