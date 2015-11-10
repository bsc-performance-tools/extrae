#include <mpi.h>
int main(int argc, char *argv[])
{
	int v;
	MPI_Request r1, r2;
	MPI_Status s1, s2;
	MPI_Init (&argc, &argv);
	MPI_Isend (&v, 1, MPI_INT, 0, 1234, MPI_COMM_WORLD, &r1);
	MPI_Irecv (&v, 1, MPI_INT, 0, 1234, MPI_COMM_WORLD, &r2);
	MPI_Wait (&r1, &s1);
	MPI_Wait (&r2, &s2);
	MPI_Finalize();
	return 0;
}
