
#
# To compile this example, set the following 4 variables.
#

XMLHOME = $(HOME)/libxml2/2.6.31/32
F77     = mpxlf
FFLAGS  = -O2
FLIBS   = -L../../../lib -lmpitracef -lpmapi -L$(XMLHOME)/lib -lxml2

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

