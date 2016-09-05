#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libcudampitrace.so
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libcudampitracef.so

## Run the desired program
$*

