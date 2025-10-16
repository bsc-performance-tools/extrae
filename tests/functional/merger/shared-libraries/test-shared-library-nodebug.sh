#!/bin/bash

source ../../helper_functions.bash

TRACE=main_nodebug

EXTRAE_ON=1 ./main_nodebug
../../../../src/merger/mpi2prv -f TRACE.mpits

# Do actual checks
CheckEntryInPCF ${TRACE}.pcf "main.c:33.*main_nodebug"

NumberEntriesInPRV ${TRACE}.prv 60000119 2
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to 60000119:2"
	exit 1
fi
NumberEntriesInPRV ${TRACE}.prv 60000119 0
if [[ "${?}" -ne 3 ]] ; then
	echo "There must be only three entries to 60000119:0"
	exit 1
fi
