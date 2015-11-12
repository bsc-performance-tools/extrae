#!/bin/bash

source ../../helper_functions.bash

TRACE=JavaException

rm -fr TRACE.* *.mpits set-0

EXTRAE_CONFIG_FILE=extrae.xml ../../../../src/launcher/java/extraej.bash -- JavaException

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual checks
CheckEntryInPCF ${TRACE}.pcf "Java exception"
CheckEntryInPCF ${TRACE}.pcf "In-flight exception"

NumberEntriesInPRV ${TRACE}.prv 48000002 1
NENTRIES=${?}
if [[ "${NENTRIES}" -lt 1 ]] ; then
	echo "There must be at least one entry to In-flight exception"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 48000002 0
NEXITS=${?}
if [[ "${NEXITS}" -ne "${NENTRIES}" ]] ; then
	echo "There must be the same number of entries to exits in Java exception (${NEXITS} / ${NENTRIES})"
	exit 1
fi

if [[ -r ${TRACE}.prv &&  -r ${TRACE}.pcf && -r ${TRACE}.row ]]; then
	rm -fr TRACE.* *.mpits set-0 ${TRACE}.pcf ${TRACE}.row ${TRACE}.prv
	exit 0
else
	die "Error checking existance for trace ${TRACE}*"
fi
