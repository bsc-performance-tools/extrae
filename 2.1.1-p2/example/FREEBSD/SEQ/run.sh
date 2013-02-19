#!/bin/sh

export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@
export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib

./pi
