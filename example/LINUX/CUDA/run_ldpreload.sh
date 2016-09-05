#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

EXTRAE_CONFIG_FILE=extrae.xml LD_PRELOAD=${EXTRAE_HOME}/lib/libcudatrace.so ./hello
