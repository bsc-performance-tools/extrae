#!/bin/tcsh

setenv MPITRACE_DIR `pwd`
setenv MPITRACE_HOME /cepba/des/gllort/mpitrace/DEC-LINUX-BGL

$MPITRACE_HOME/bin/mpitrace /bin/dmpirun -np 2 /cepba/des/gllort/mpitrace/DEC-LINUX-BGL/example/DEC/mpi_ping


