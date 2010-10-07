MPI_HOME = @sub_MPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@

F77 = $(MPI_HOME)/bin/mpif77 
FFLAGS = -O2

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

