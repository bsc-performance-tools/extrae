#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export OMP_NUM_THREADS=1
export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/liboacccudatrace.so # For C apps
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitracef.so # For Fortran apps

## Run the desired program
$*

