#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXECUTABLE=./pi

${EXTRAE_HOME}/bin/extrae -config extrae.xml ${EXECUTABLE}
