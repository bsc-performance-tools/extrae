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

#include <mpi.h>
int main(int argc, char *argv[])
{
	int v, rank, size;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	MPI_Comm_size (MPI_COMM_WORLD, &size);
	if (size != 2)
		return 1;
	if (rank == 0)
	{
		MPI_Send (&v, 1, MPI_INT, 1, 1234, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Request r;
		MPI_Status s;
		int flag = 0;
		MPI_Iprobe (0, 1234, MPI_COMM_WORLD, &flag, &s);
		while (!flag)
			MPI_Iprobe (0, 1234, MPI_COMM_WORLD, &flag, &s);
		MPI_Irecv (&v, 1, MPI_INT, 0, 1234, MPI_COMM_WORLD, &r);
		MPI_Wait (&r, &s);
	}
	MPI_Finalize();
	return 0;
}
