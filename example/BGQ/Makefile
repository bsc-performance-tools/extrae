
BG_HOME = /bgsys/drivers/ppcfloor
PAPI_HOME = @sub_PAPI_HOME@
MPI_HOME = @sub_MPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@
XML2_LIBS = 

F77 = $(MPI_HOME)/bin/mpif77
FLIBS = -L$(EXTRAE_HOME)/lib -lmpitracef -L$(PAPI_HOME)/lib @sub_PAPI_LIBS@
FFLAGS = -O2 -g

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o TRACE*.* pingtmp?

