#!/bin/bash

source ../helper_functions.bash

TRACENAME=trace.prv EXTRAE_CONFIG_FILE=extrae_envvar_merge.xml ./check_Extrae_xml

rm -fr set-0 TRACE.*

if [[ -f trace.prv ]] ; then
	rm -fr trace.???
	exit 0
else
	exit 1
fi

