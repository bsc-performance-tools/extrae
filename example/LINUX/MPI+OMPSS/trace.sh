#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=extrae.xml
export NX_ARGS="${NX_ARGS} --instrumentation=extrae "
export LD_PRELOAD=${EXTRAE_HOME}/lib/libnanosmpitrace.so
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libnanosmpitracef.so

$*
