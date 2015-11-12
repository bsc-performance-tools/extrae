#!/bin/bash

source ../../helper_functions.bash

TRACE=JavaSimple

rm -fr TRACE.* *.mpits set-0

EXTRAE_CONFIG_FILE=extrae.xml ../../../../src/launcher/java/extraej.bash -- JavaSimple

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

if [[ -r ${TRACE}.prv &&  -r ${TRACE}.pcf && -r ${TRACE}.row ]]; then
	rm -fr TRACE.* *.mpits set-0 ${TRACE}.pcf ${TRACE}.row ${TRACE}.prv
	exit 0
else
	die "Error checking existance for trace ${TRACE}*"
fi
