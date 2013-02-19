#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@
${EXTRAE_HOME}/bin/mpi2prv -e pi_f -f TRACE.mpits
