#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

EXTRAE_CONFIG_FILE=extrae.xml LD_LIBRARY_PATH=${EXTRAE_HOME}/lib: ./hello_instrumented
