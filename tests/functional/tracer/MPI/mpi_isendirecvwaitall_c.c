#include <mpi.h>
int main(int argc, char *argv[])
{
	int v;
	MPI_Request r[2];
	MPI_Status s[2];
	MPI_Init (&argc, &argv);
	MPI_Isend (&v, 1, MPI_INT, 0, 1234, MPI_COMM_WORLD, &r[0]);
	MPI_Irecv (&v, 1, MPI_INT, 0, 1234, MPI_COMM_WORLD, &r[1]);
	MPI_Waitall (2, r, s);
	MPI_Finalize();
	return 0;
}
