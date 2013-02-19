include ../../Makefile.inc

NVCC = nvcc
CFLAGS = -O2 -g -I$(MPI_HOME)/include
CLIBS = -L$(MPI_HOME)/lib -lmpi

all: mpi_hello 

mpi_hello: mpi_hello.cu
	$(NVCC) $(CFLAGS) mpi_hello.cu $(CLIBS) -o mpi_hello

clean:
	rm -fr mpi_hello TRACE.mpits TRACE.sym set-0

