#!/bin/bash

export EXTRAE_CONFIG_FILE=extrae-instrument-functions.xml
export EXTRAE_HOME=@sub_PREFIXDIR@
export LD_PRELOAD=${EXTRAE_HOME}/lib/libseqtrace.so

./pi_instrumented_functions
