#!/bin/sh

export MPITRACE_HOME=@sub_PREFIXDIR@
export MPTRACE_CONFIG_FILE=mpitrace.xml
export LD_PRELOAD=${MPITRACE_HOME}/lib/libmpitrace.so

## Run the desired program
$*

