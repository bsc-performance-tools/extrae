#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

${EXTRAE_HOME}/bin/mpi2prv -s TRACE.sym -f TRACE.mpits -e mpi_ping -o mpi_ping.prv

