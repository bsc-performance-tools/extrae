#!/bin/tcsh

setenv EXTRAE_HOME @sub_PREFIXDIR@
${EXTRAE_HOME}/bin/mpi2prv -f *.mpits -o mpi_ping.prv -e $name

