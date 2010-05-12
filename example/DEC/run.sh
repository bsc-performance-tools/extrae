#!/bin/tcsh

setenv MPITRACE_HOME @sub_PREFIXDIR@

${MPITRACE_HOME}/bin/mpitrace /bin/dmpirun -np 2 /cepba/des/gllort/mpitrace/DEC-LINUX-BGL/example/DEC/mpi_ping


