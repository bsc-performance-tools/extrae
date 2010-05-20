#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=mpitrace.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so

## Run the desired program
$*

