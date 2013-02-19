#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:@sub_MPI_HOME@/lib:@sub_PAPI_HOME@/lib

## Run the desired program
$*

