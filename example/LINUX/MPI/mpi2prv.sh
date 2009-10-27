#!/bin/tcsh

set name=mpi_ping

setenv MPITRACE_HOME /gpfs/apps/CEPBATOOLS/64.hwc
$MPITRACE_HOME/bin/mpi2prv -f TRACE.mpits -o $name.prv

