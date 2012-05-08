#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=../extrae.xml
source ${EXTRAE_HOME}/etc/extrae.sh

## Run the desired program
$*

