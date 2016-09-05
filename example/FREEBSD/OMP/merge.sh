#!/bin/sh

source @sub_PREFIXDIR@/etc/extrae.sh

${EXTRAE_HOME}/bin/mpi2prv -e pi_f -f TRACE.mpits
