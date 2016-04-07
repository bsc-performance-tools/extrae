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

#include "mpi2out.h"

#if defined(PARALLEL_MERGE)

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif

#include <mpi.h>
#include "mpi-aux.h"

int main (int argc, char *argv[])
{
	int res;
	int ntasks;
	int idtask;

	res = MPI_Init (&argc, &argv);
	MPI_CHECK (res, MPI_Init, "Failed to initialize MPI");

	res = MPI_Comm_size (MPI_COMM_WORLD, &ntasks);
	MPI_CHECK (res, MPI_Comm_size, "Failed to call MPI_Comm_size");

	res = MPI_Comm_rank (MPI_COMM_WORLD, &idtask);
	MPI_CHECK (res, MPI_Comm_size, "Failed to call MPI_Comm_rank");

	merger_pre (ntasks);
	ProcessArgs (idtask, argc, argv);
	merger_post (ntasks, idtask);

	res = MPI_Finalize ();
	MPI_CHECK (res, MPI_Finalize, "Failed to uninitialize MPI");

	return 0;
}
#else
int main (int argc, char *argv[])
{
	merger_pre (1);
	ProcessArgs (0, argc, argv);
	merger_post (1, 0);

	return 0;
}
#endif

