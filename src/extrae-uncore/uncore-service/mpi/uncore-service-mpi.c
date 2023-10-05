#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

//#define DEBUG

#if defined(DEBUG)
# define debug_msg(stream, fmt, ...) fprintf(stream, fmt, ##__VA_ARGS__)
#else
# define debug_msg(stream, fmt, ...)
#endif

int main(int argc, char **argv)
{
	int rank = 0, size = 0, stop = 0;

	debug_msg(stderr, "[DEBUG] UNCORE SERVICE INITIATES\n");

	MPI_Init(&argc, &argv);

	PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
	PMPI_Comm_size(MPI_COMM_WORLD, &size);

	MPI_Comm intercomm;
        PMPI_Comm_get_parent(&intercomm);

	debug_msg(stderr, "[DEBUG] UNCORE SERVICE BLOCKS IN BCAST\n");
	PMPI_Bcast(&stop, 1, MPI_INT, 0, intercomm);
	debug_msg(stderr, "[DEBUG] UNCORE SERVICE WAKES UP FROM BCAST\n");

	debug_msg(stderr, "[DEBUG] UNCORE READER STOPPING\n");
	MPI_Finalize();
	debug_msg(stderr, "[DEBUG] UNCORE READER QUITS\n");

	return 0;
}
