#!/bin/sh

export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=/home/harald/T
export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib

./pi_instrumented

${EXTRAE_HOME}/bin/mpi2prv -e pi_instrumented -f TRACE.mpits -o pi.prv
