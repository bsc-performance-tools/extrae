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

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>

void do_work (int rank, int size)
{
	int i, n;
	unsigned lowindex, highindex;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, tmp, area, x;

	if (rank == 0)
	{
		n = size*1000*1000;
		assert (
		  MPI_Bcast (&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD)
		    == MPI_SUCCESS
		);
	}
	else
	{
		assert (
		  MPI_Bcast (&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD)
		    == MPI_SUCCESS
		);
	}

	h = 1.0 / (double) n;
	tmp = 0.0;
	lowindex = (n/size)*rank;
	highindex = (n/size)*(rank+1)-1;

	#pragma omp parallel for private(x) reduction(+:tmp)
	for (i = lowindex; i <= highindex; i++)
	{
		x = h * ((double)i - 0.5);
		tmp += (4.0 / (1.0 + x*x));
	}

	assert (
	  MPI_Reduce (&tmp, &area, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD)
	    == MPI_SUCCESS
	);

	if (rank == 0)
	{
		pi = h * area;
		printf("pi (by using #pragma omp parallel for) is approximately %.16f, Error is %.16f\n",pi,fabs(pi - PI25DT));
	}
}

int main(int argc, char **argv)
{
	int size, rank;

	assert (MPI_Init (&argc, &argv) == MPI_SUCCESS);
	assert (MPI_Comm_size (MPI_COMM_WORLD, &size) == MPI_SUCCESS);
	assert (MPI_Comm_rank (MPI_COMM_WORLD, &rank) == MPI_SUCCESS);
	do_work (rank, size);
	assert (MPI_Finalize () == MPI_SUCCESS);
}
