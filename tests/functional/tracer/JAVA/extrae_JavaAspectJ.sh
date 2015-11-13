#!/bin/bash

source ../../helper_functions.bash

echo JavaFunction.run > user-functions

TRACE=JavaFunction

rm -fr TRACE.* *.mpits set-0

EXTRAE_CONFIG_FILE=extrae-function.xml ../../../../src/launcher/java/extraej.bash -- JavaFunction

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

rm user-functions

# Actual checks

CheckEntryInPCF ${TRACE}.pcf "User function line"
 
NumberEntriesInPRV ${TRACE}.prv 60000019 3
if [[ "${?}" -ne 1 ]] ; then
	die "There must be at least one entry to JavaFunction.run (function)"
fi

NumberEntriesInPRV ${TRACE}.prv 60000019 0
if [[ "${?}" -ne 1 ]] ; then
	die "There must be at least one exit to JavaFunction.run (function)"
fi

NumberEntriesInPRV ${TRACE}.prv 60000119 3
if [[ "${?}" -ne 1 ]] ; then
	die "There must be at least one entry to JavaFunction.java (file-name)"
fi

NumberEntriesInPRV ${TRACE}.prv 60000119 0
if [[ "${?}" -ne 1 ]] ; then
	die "There must be at least one entry to JavaFunction.java (file-name)"
fi

if [[ -r ${TRACE}.prv &&  -r ${TRACE}.pcf && -r ${TRACE}.row ]]; then
	rm -fr TRACE.* *.mpits set-0 ${TRACE}.pcf ${TRACE}.row ${TRACE}.prv
	exit 0
else
	die "Error checking existance for trace ${TRACE}*"
fi
