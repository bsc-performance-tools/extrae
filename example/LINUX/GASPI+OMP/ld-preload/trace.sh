#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

export OMP_NUM_THREADS=2
export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libompgaspitrace.so

## Run the desired program
$*
