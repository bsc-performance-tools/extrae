#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so

## Run the desired program
$*

