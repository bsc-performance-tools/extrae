include ../Makefile.inc

CFLAGS = -O2 -g $(MPI_CFLAGS)
CLIBS = -O2 -g $(MPI_LDFLAGS) $(MPI_LIBS)

all: mpi_hello 

mpi_hello: mpi_hello.cu
	$(NVCC) $(CFLAGS) mpi_hello.cu $(CLIBS) -o mpi_hello

clean:
	rm -fr mpi_hello TRACE.mpits TRACE.sym set-0

