
#
# To compile this example, set the following 4 variables.
#

XMLHOME = $(HOME)/libxml2/2.6.31/32
EXTRAE_HOME = @sub_PREFIXDIR@
MPI_HOME= @sub_MPI_HOME@
FFLAGS  = -O2 -bmaxdata:0x7fffffff
FLIBS   = -L$(EXTRAE_HOME)/lib -lmpitracef -lpmapi -L$(XMLHOME)/lib -lxml2 @sub_BFD_LIBS@ @sub_LIBERTY_LIBS@
F77     = $(MPI_HOME)/mpxlf

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

