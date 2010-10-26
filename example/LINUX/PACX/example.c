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

#include <pacx.h>
#include <stdio.h>
#include <math.h>

/*
  This example is based on the example that can be found in

  http://www.mcs.anl.gov/research/projects/mpi/usingmpi/examples/simplempi/main.htm

  To run:
	echo "localhost 4" > ~/.hostfile
  mpirun -np 6 [4+2] ./BINARY
*/


int main (int argc, char *argv[])
{
	int NSTEPS;
	int res, rank, size, i;
	double PI25DT = 3.141592653589793238462643;
	double mypi, pi, h, sum, x;

	res = PACX_Init (&argc, &argv);

	res = PACX_Comm_rank (PACX_COMM_WORLD, &rank);
	res = PACX_Comm_size (PACX_COMM_WORLD, &size);

	/* Distribute NSTEPS, a fixed value or bcast could work, but try on send/recv */
	if (rank == 0)
	{
		int j;
		NSTEPS = 10000000;
		for (j = 1; j < size; j++)
			res = PACX_Send (&NSTEPS, 1, PACX_INT, j, 123456, PACX_COMM_WORLD);
	}
	else
	{
		PACX_Status s;
		res = PACX_Recv (&NSTEPS, 1, PACX_INT, 0, 123456, PACX_COMM_WORLD, &s);
	}
	
	/* Do the computation */
	h   = 1.0 / (double) NSTEPS; 
	sum = 0.0; 
	for (i = rank + 1; i <= NSTEPS; i += size)
	{ 
		x = h * ((double)i - 0.5); 
		sum += (4.0 / (1.0 + x*x)); 
	} 
	mypi = h * sum; 

	res = PACX_Reduce (&mypi, &pi, 1, PACX_DOUBLE, PACX_SUM, 0, PACX_COMM_WORLD); 
	if (rank == 0)  
		printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT)); 
	
	res = PACX_Barrier (PACX_COMM_WORLD);
	res = PACX_Finalize ();
	return 0;
}
