
BG_HOME	= /bgsys/drivers/ppcfloor
PAPI_HOME	= /bgsys/local/papi/papi-c-3.9.0
MPITRACE_HOME	= ../../..
XML2_LIBS       = 

F77		= $(BG_HOME)/comm/bin/mpif77
FLIBS		= -L$(MPITRACE_HOME)/lib -lmpitracef -L$(PAPI_HOME)/lib -lpapi
FFLAGS		= -O2


all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o TRACE*.* pingtmp?

