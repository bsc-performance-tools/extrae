#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

EXTRAE_CONFIG_FILE=extrae.xml LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:${LD_LIBRARY_PATH} ./hello_instrumented
