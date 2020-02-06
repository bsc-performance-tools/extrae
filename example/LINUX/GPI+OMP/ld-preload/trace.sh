#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export OMP_NUM_THREADS=2
export EXTRAE_CONFIG_FILE=@sub_PREFIXDIR@/share/example/GPI+OMP/ld-preload/../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libogaspitrace.so

## Run the desired program
$*
