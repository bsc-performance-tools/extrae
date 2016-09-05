#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXTRAE_CONFIG_FILE=extrae.xml
export EXECUTABLE=./pi_instrumentedf # This is taken by extrae.xml

${EXECUTABLE}
