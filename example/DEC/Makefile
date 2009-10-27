MPITRACE_HOME	= /cepba/des/gllort/mpitrace/DEC-LINUX-BGL

FLIBS		= -L$(MPITRACE_HOME)/lib -lmpitracef -lpthread -lrt -lmach -lpmpi -lmpi -lexc
F77		= f77
FFLAGS		= -O2 -g


all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o

