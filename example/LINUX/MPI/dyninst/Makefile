include ../../Makefile.inc

F77 = $(MPI_HOME)/bin/mpif77 
FFLAGS = -O2 -g
FLIBS = 

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -fr mpi_ping *.o pingtmp? TRACE.* set-*

