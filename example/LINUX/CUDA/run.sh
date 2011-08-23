#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@
export PAPI_HOME=@sub_PAPI_HOME@

EXTRAE_CONFIG_FILE=extrae.xml LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:${PAPI_HOME}/lib:${LD_LIBRARY_PATH} ./hello
${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -e ./hello
