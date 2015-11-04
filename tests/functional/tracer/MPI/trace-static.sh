#!/bin/sh

TOP_BUILDDIR=../../../../

export EXTRAE_HOME=${TOP_BUILDDIR}
export EXTRAE_CONFIG_FILE=extrae.xml 

$*

