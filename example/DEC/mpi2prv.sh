#!/bin/tcsh

set name=mpi_ping

setenv MPITRACE_HOME $HOME/mpitrace/DEC-LINUX-BGL
$MPITRACE_HOME/bin/mpi2prv -f *.mpits -o $name.prv -e $name

