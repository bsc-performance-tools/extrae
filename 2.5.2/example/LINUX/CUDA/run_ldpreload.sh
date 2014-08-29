#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

EXTRAE_CONFIG_FILE=extrae.xml LD_PRELOAD=${EXTRAE_HOME}/lib/libcudatrace.so ./hello
