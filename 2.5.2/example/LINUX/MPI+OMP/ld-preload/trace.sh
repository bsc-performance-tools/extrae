#!/bin/bash

#Workaround for tracing in MN3, make TMPDIR point to an existing dir
if [ ! -z "${TMPDIR}" ]; then
	export TMPDIR=$TMPDIR/extrae
	mkdir -p $TMPDIR
fi

export EXTRAE_HOME=@sub_PREFIXDIR@
export EXTRAE_CONFIG_FILE=../extrae.xml
export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitrace.so # For C apps
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libompitracef.so # For Fortran apps

## Run the desired program
$*

