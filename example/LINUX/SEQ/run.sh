#!/bin/sh

export MPTRACE_CONFIG_FILE=mpitrace.xml
export MPITRACE_HOME=@sub_PREFIXDIR@
export LD_LIBRARY_PATH=${MPITRACE_HOME}/lib

./pi
