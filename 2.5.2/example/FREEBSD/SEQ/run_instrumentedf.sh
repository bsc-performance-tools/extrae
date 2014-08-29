#!/bin/sh

export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=@sub_PREFIXDIR@
source ${EXTRAE_HOME}/etc/extrae.sh

./pi_instrumentedf
