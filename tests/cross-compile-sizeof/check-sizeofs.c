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
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <mpi.h>

#define PRINT_SIZE(t) \
  printf ("sizeof(%s) = %ld\n", #t, sizeof(t));

int main (int argc, char *argv[])
{
	MPI_Status s;
	long addr1 = (long) &s;
	long addr2 = (long) &(s.MPI_SOURCE);
	long addr3 = (long) &(s.MPI_TAG);

	PRINT_SIZE(long long)
	PRINT_SIZE(long)
	PRINT_SIZE(int)
	PRINT_SIZE(pid_t)
	PRINT_SIZE(ssize_t)
	PRINT_SIZE(size_t)
	PRINT_SIZE(void*)
	PRINT_SIZE(short)
	PRINT_SIZE(char)
	
	printf ("size for MPI_Status in sizeof(int) = %ld\n", sizeof(MPI_Status)/sizeof(int));
	printf ("offset of MPI_Status.MPI_SOURCE in sizeof(int) = %ld\n", (addr2-addr1)/sizeof(int));
	printf ("offset of MPI_Status.MPI_TAG in sizeof(int) = %ld\n", (addr3-addr1)/sizeof(int));

	return 0;
}

