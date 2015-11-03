include ../Makefile.inc

FFLAGS = -O2 -g
FLIBS = $(EXTRAE_HOME)/lib/libmpitracef.a $(PAPI_LIBS) $(BFD_LIBS) $(XML2_LIBS) $(UNWIND_LIBS) $(POSIX_CLOCK_LIBS) $(LIBEXECINFO_LIBS)

all: mpi_ping

mpi_ping: mpi_ping.f
	$(MPIF77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

