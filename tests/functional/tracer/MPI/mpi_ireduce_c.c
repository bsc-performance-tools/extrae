#include <mpi.h>
int main(int argc, char *argv[])
{
	int v, vv;
	MPI_Request r;
	MPI_Status s;
	MPI_Init (&argc, &argv);
	MPI_Ireduce (&v, &vv, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD, &r);
	MPI_Wait (&r, &s);
	MPI_Finalize();
	return 0;
}
