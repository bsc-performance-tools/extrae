MPI_HOME = @sub_MPI_HOME@
EXTRAE_HOME = @sub_PREFIXDIR@
PAPI_HOME = @sub_PAPI_HOME@
XML2_LDFLAGS = @sub_XML2_LDFLAGS@
XML2_LIBS = @sub_XML2_LIBS@

F77 = $(MPI_HOME)/bin/mpif77 
FFLAGS = -O2 -g
FLIBS = $(EXTRAE_HOME)/lib/libmpitracef.a -L$(PAPI_HOME)/lib @sub_PAPI_LIBS@ $(XML2_LDFLAGS) $(XML2_LIBS)  @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

