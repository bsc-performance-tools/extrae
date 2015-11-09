#include <mpi.h>
int main(int argc, char *argv[])
{
	int v, vv;
	MPI_Init (&argc, &argv);
	MPI_Reduce (&v, &vv, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
