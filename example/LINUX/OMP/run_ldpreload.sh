#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export OMP_NUM_THREADS=4
export EXTRAE_CONFIG_FILE=extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so

./pi
