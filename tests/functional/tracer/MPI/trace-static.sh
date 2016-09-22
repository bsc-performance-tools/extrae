#!/bin/bash

TOP_BUILDDIR=../../../../

export EXTRAE_HOME=${TOP_BUILDDIR}
source ${EXTRAE_HOME}/etc/extrae.sh
export EXTRAE_CONFIG_FILE=extrae.xml 

$*

