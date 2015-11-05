#include <mpi.h>
int main(int argc, char *argv[])
{
	MPI_Request r;
	MPI_Status s;
	MPI_Init (&argc, &argv);
	MPI_Ibarrier (MPI_COMM_WORLD, &r);
	MPI_Wait (&r, &s);
	MPI_Finalize();
	return 0;
}
