#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NITERS 100

int main(int argc, char ** argv) {
	int i, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	for (i=0; i<NITERS; i++) 
	{

        	MPI_Barrier (MPI_COMM_WORLD);
		sleep(1);
	}

	MPI_Finalize();
}

