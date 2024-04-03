include ../Makefile.inc

CFLAGS = -O2 -g --generate-line-info --cudart shared $(MPI_CFLAGS)
CLIBS  = $(MPI_LDFLAGS) $(MPI_LIBS)

all: mpi_hello 

mpi_hello: mpi_hello.cu
	$(NVCC) $(CFLAGS) mpi_hello.cu $(CLIBS) -o mpi_hello

clean:
	rm -fr mpi_hello TRACE.mpits TRACE.sym set-0

