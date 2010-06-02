#!/bin/sh

export MPI_HOME=@sub_MPI_HOME@
export EXTRAE_CONFIG_FILE=@sub_PREFIXDIR@

${MPI_HOME}/bin/mpirun -np 2 ./mpi_ping

