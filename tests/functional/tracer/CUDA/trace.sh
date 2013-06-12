#!/bin/sh

TOP_BUILDDIR=../../../../

export EXTRAE_HOME=${TOP_BUILDDIR}
export EXTRAE_CONFIG_FILE=extrae.xml 
export LD_PRELOAD=${TOP_BUILDDIR}/src/tracer/.libs/libcudatrace.so

$*

