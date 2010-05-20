#!/bin/tcsh

setenv EXTRAE_HOME @sub_PREFIXDIR@

${EXTRAE_HOME}/bin/mpitrace /bin/dmpirun -np 2 /cepba/des/gllort/mpitrace/DEC-LINUX-BGL/example/DEC/mpi_ping
