MPI_HOME = @sub_MPI_HOME@
MPITRACE_HOME = @sub_PREFIXDIR@
PAPI_HOME = @sub_PAPI_HOME@

F77 = $(MPI_HOME)/bin/mpif77 
FFLAGS = -O2
FLIBS = -L$(PAPI_HOME)/lib @sub_PAPI_LIBS@

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

