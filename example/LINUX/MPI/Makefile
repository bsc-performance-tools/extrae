
#
# To compile this example, set the following 4 variables.
#

MPICH_HOME	  = /opt/osshpc/mpich-gm/64
MPITRACE_HOME = /gpfs/apps/CEPBATOOLS/64.hwc
PAPI_HOME	  = /gpfs/apps/PAPI/papi-3.2.1/64
PERFCTR_HOME  = /gpfs/apps/PAPI/papi-3.2.1/64

F77     = $(MPICH_HOME)/bin/mpif77 
FFLAGS  = -O2

all: mpi_ping

mpi_ping: mpi_ping.f
	$(F77) $(FFLAGS) mpi_ping.f $(FLIBS) -o mpi_ping

clean:
	rm -f mpi_ping *.o pingtmp? TRACE.*

