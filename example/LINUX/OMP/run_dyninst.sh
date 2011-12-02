#!/bin/sh

export OMP_NUM_THREADS=4
export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@

${EXTRAE_HOME}/bin/extrae -v ./pi

#${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -s TRACE.sym -e ./pi -o pi.prv
