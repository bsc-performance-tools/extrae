include ../Makefile.inc

FFLAGS = -O2 -g

all: mpi_ping

mpi_ping: mpi_ping.f
	$(MPIF77) $(FFLAGS) mpi_ping.f -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

