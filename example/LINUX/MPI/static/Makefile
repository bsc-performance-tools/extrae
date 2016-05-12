include ../Makefile.inc

FFLAGS = -O2 -g
FLIBS = $(EXTRAE_HOME)/lib/libmpitracef.a $(UNWIND_LIBS) $(PAPI_LIBS) $(XML2_LDFLAGS) $(XML2_LIBS) $(BFD_LIBS) $(POSIX_CLOCK_LIBS) $(ZLIB_LIBS) $(DL_LIBS)

all: mpi_ping

mpi_ping: mpi_ping.f
	$(MPIF77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -fr mpi_ping *.o pingtmp? TRACE.* set-*

