#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@
source ${EXTRAE_HOME}/etc/extrae.sh
export EXECUTABLE=./pi # This is taken by extrae.xml

${EXTRAE_HOME}/bin/extrae -config extrae.xml ${EXECUTABLE}
