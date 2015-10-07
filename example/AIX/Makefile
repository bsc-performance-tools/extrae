include Makefile.inc

XMLHOME = $(HOME)/libxml2/2.6.31/32
FFLAGS  = -O2 -bmaxdata:0x7fffffff

FLIBS   = -L$(EXTRAE_HOME)/lib -lmpitracef $(XML2_LDFLAGS) $(XML2_LIBS) $(BFD_LIBS) $(PAPI_LIBS) -lpmapi
F77     = $(MPIF77)

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

