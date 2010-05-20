#!/bin/sh

export EXTRAE_ON=1
export EXTRAE_HOME=@sub_PREFIXDIR@
export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib
./example

