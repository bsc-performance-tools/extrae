#!/bin/sh

export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@

./pi_instrumented

${EXTRAE_HOME}/bin/mpi2prv -e pi_instrumented -f TRACE.mpits -o pi.prv
