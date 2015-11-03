#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXECUTABLE=./pi

${EXTRAE_HOME}/bin/extrae -config extrae.xml ${EXECUTABLE}
