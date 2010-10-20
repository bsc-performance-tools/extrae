EXTRAE_HOME = @sub_PREFIXDIR@ 
MPI_HOME = @sub_MPI_HOME@

FLIBS = -L$(EXTRAE_HOME)/lib -lmpitracef -lpthread -lrt -lmach -lpmpi -lmpi -lexc @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@
F77	= $(MPI_HOME)/bin/mpif77
FFLAGS = -O2 -g

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o

