#!/bin/tcsh

setenv MPITRACE_HOME @sub_PREFIXDIR@
$MPITRACE_HOME/bin/mpi2prv -f *.mpits -o mpi_ping.prv -e $name

