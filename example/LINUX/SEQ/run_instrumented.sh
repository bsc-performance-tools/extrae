#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXTRAE_CONFIG_FILE=extrae.xml
export EXECUTABLE=./pi_instrumented 

${EXECUTABLE}
