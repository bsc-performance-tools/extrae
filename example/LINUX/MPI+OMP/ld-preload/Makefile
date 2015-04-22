include ../../Makefile.inc

MPICC = $(MPI_HOME)/bin/mpicc
CFLAGS = -O2 -g -fopenmp

all: pi

pi: pi.c
	$(MPICC) $(CFLAGS) pi.c -o pi

clean:
	rm -fr pi TRACE.* set-?

