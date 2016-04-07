#!/bin/bash

export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@
export EXECUTABLE=./pi_instrumented 

${EXECUTABLE}
