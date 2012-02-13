#!/bin/bash

export OMP_NUM_THREADS=4
export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@
source ${EXTRAE_HOME}/etc/extrae.sh

LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so ./pi_instrumented
