#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXTRAE_CONFIG_FILE=extrae-instrument-functions.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libseqtrace.so

./pi_instrumented_functions
