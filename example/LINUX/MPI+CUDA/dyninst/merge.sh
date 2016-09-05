#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

${EXTRAE_HOME}/bin/mpi2prv -s TRACE.sym -f TRACE.mpits -e mpi_ping -o mpi_ping.prv

