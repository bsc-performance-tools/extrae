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
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NITERS 150

/**
 * Test MPI program where every task sends a message to the next, 
 * then all tasks synchronize and sleep for 1 second. 
 *
 * Beyond iteration #100, the length of the iteration is doubled, 
 * so that the analysis detects two different periodic behaviors
 * and takes a sample for each of them.
 */
int main(int argc, char ** argv) 
{
  int i, rank, size, from, to;
  int buf;
  int factor = 1;
  MPI_Request req1, req2;
  MPI_Status sts1, sts2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);

  from = rank - 1;
  if (from < 0) from = size - 1; 
  to = rank + 1;
  if (to >= size) to = 0;

  for (i=0; i<NITERS; i++) 
  {
    if ((rank == 0) && (i % 10 == 0) && (i > 0)) 
    {
      fprintf(stdout, "# Iteration %d\n", i);
      fflush(stdout);
    }
    if (i == 100) factor ++;

    MPI_Isend(&buf, 1, MPI_INT, to, 0, MPI_COMM_WORLD, &req1);
    MPI_Irecv(&buf, 1, MPI_INT, from, 0, MPI_COMM_WORLD, &req2);

    usleep(250000 * factor);

    MPI_Wait(&req1, &sts1);
    MPI_Wait(&req2, &sts2);
  
    usleep(500000 * factor);

    MPI_Barrier (MPI_COMM_WORLD);

    usleep(750000 * factor);
  }

  MPI_Finalize();
}

