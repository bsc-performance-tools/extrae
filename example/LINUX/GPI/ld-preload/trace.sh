#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXTRAE_CONFIG_FILE=@sub_PREFIXDIR@/share/example/GPI/ld-preload/../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libgaspitrace.so

## Run the desired program
$*
