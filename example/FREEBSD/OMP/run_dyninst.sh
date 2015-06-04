#!/bin/sh

export OMP_NUM_THREADS=4
export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@

${EXTRAE_HOME}/bin/extrae -v ./pi
