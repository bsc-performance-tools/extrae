#!/bin/tcsh

setenv MPI_HOME @sub_MPI_HOME@
setenv EXTRAE_HOME @sub_PREFIXDIR@
setenv EXTRAE_COUNTERS 0x80000007,0x80000032

${MPI_HOME}/bin/mpirun -np 2 ./mpi_ping

