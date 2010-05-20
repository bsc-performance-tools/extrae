#!/bin/sh

export MPITRACE_ON=1
export MPITRACE_HOME=@sub_PREFIXDIR@
export LD_LIBRARY_PATH=${MPITRACE_HOME}/lib
./example

