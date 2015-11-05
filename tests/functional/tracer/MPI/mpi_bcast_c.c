#include <mpi.h>
int main(int argc, char *argv[])
{
	int v;
	MPI_Init (&argc, &argv);
	MPI_Bcast (&v, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
