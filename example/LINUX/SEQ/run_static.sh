#!/bin/sh

export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@
export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:@sub_MPI_HOME@/lib:@sub_PAPI_HOME@/lib:@sub_UNWIND_HOME@/lib

./pi_instrumented

${EXTRAE_HOME}/bin/mpi2prv -e pi_instrumented -f TRACE.mpits -o pi.prv
