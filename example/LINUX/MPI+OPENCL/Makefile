include Makefile.inc

CFLAGS= -g -I$(OPENCL_HOME)/include -O3 -lm -Wall
LIBS = $(OPENCL_LDFLAGS) -lOpenCL

mpi-vadd: mpi-vadd.c 
	$(MPICC) $^ $(CFLAGS) $(LIBS) -o $@

clean:
	rm -fr mpi-vadd set-0 TRACE.mpits TRACE.sym
