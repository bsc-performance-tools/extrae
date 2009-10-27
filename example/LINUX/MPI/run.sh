#!/bin/tcsh

setenv MPICH_HOME /home/tools/mpich
setenv MPTRACE_DIR `pwd`
setenv MPITRACE_HOME /home/cepba/fescale/tools/mpitrace/DEC-LINUX
setenv MPTRACE_COUNTERS 0x80000007,0x80000032

$MPICH_HOME/bin/mpirun -np 2 ./mpi_ping


