#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=./extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so    # C programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so  # Fortran programs

## Run the program
$*

