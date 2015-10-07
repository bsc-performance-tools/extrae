include Makefile.inc

FLIBS = -L$(EXTRAE_HOME)/lib -lmpitracef -L$(PAPI_HOME)/lib $(PAPI_LIBS)
FFLAGS = -O2 -g

all: mpi_ping

mpi_ping: mpi_ping.f
	$(MPIF77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o TRACE*.* pingtmp?

