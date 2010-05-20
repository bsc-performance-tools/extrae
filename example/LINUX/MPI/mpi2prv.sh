#!/bin/tcsh

setenv MPITRACE_HOME @sub_PREFIXDIR@
${MPITRACE_HOME}/bin/mpi2prv -f TRACE.mpits -o mpi_ping.prv

