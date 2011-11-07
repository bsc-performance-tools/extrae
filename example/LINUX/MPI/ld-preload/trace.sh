#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:@sub_MPI_HOME@/lib:@sub_PAPI_HOME@/lib:@sub_UNWIND_HOME@/lib
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so

## Run the desired program
$*

