#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXTRAE_CONFIG_FILE=../extrae.xml
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so # For C apps
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so # For Fortran apps

## Run the desired program
$*

