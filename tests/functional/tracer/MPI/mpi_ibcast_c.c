#include <mpi.h>
int main(int argc, char *argv[])
{
	int v;
	MPI_Request r;
	MPI_Status s;
	MPI_Init (&argc, &argv);
	MPI_Ibcast (&v, 1, MPI_INT, 0, MPI_COMM_WORLD, &r);
	MPI_Wait (&r, &s);
	MPI_Finalize();
	return 0;
}
