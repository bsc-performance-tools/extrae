
BGL_HOME	= /bgl/BlueLight/ppcfloor
PAPI_HOME	= /bgl/local/packages/papi-3.5.0
MPITRACE_HOME	= ../..
XML2_LIBS       = -L/bgl/local/packages/libxml2-2.6.20/lib -lxml2 -L/bgl/BlueLight/ppcfloor/blrts-gnu/powerpc-bgl-blrts-gnu/lib -lc -lnss_dns -lnss_files -lresolv

F77		= blrts_f77
FLIBS		= -L$(MPITRACE_HOME)/lib -lmpitracef -L$(PAPI_HOME)/lib -lpapi -L$(BGL_HOME)/bglsys/lib -lbgl_perfctr.rts -lmpich.rts -lmsglayer.rts -ldevices.rts -lrts.rts $(XML2_LIBS)
FFLAGS		= -O2 -I$(BGL_HOME)/bglsys/include


all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o TRACE*.* pingtmp?

