#!/bin/sh

export OMP_NUM_THREADS=4
export MPTRACE_CONFIG_FILE=mpitrace.xml
export MPITRACE_HOME=@sub_PREFIXDIR@
source ${MPITRACE_HOME}/etc/ompitrace.sh

${MPITRACE_HOME}/bin/ompitrace -v ./pi_f
