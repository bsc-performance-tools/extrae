#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libgaspitrace.so

## Run the desired program
$*
