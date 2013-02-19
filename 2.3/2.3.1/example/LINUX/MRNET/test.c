#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NITERS 100000

unsigned long long do_work()
{
	int i;
	unsigned long long res = 0;
 
	for (i=0; i<100000000; i++)
	{
		res += i * 2.77 / 3.14;
	}	

	return res;
}

int main(int argc, char ** argv) {
	int i, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	for (i=0; i<NITERS; i++) {

		if ((i % 2) == 0) {
			sleep(1);
		}
		else {
			unsigned long long res = do_work();
			if (rank == 0) fprintf(stderr, "Step %d: %lld\n", i, res);
		}
	
        MPI_Barrier (MPI_COMM_WORLD);
	}

	MPI_Finalize();
}

