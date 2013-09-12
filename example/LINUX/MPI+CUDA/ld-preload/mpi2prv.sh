#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@
${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o mpi_ping.prv
