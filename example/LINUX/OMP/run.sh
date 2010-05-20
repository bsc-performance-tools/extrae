#!/bin/sh

export OMP_NUM_THREADS=4
export EXTRAE_CONFIG_FILE=mpitrace.xml
export EXTRAE_HOME=@sub_PREFIXDIR@
source ${EXTRAE_HOME}/etc/ompitrace.sh

${EXTRAE_HOME}/bin/ompitrace -v ./pi_f
