#!/bin/bash

source ../../helper_functions.bash

TRACE=main

make -f Makefile.nodebug clean run

# Do actual checks
CheckEntryInPCF ${TRACE}.pcf "0 (Unresolved)"
CheckEntryInPCF ${TRACE}.pcf "0 (_NOT_Found)"
CheckEntryInPCF ${TRACE}.pcf "main.c, main"

NumberEntriesInPRV ${TRACE}.prv 60000119 3
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to 60000119:3"
	exit 1
fi
NumberEntriesInPRV ${TRACE}.prv 60000119 1
if [[ "${?}" -ne 2 ]] ; then
	echo "There must be only two entries to 60000119:1"
	exit 1
fi
NumberEntriesInPRV ${TRACE}.prv 60000119 0
if [[ "${?}" -ne 3 ]] ; then
	echo "There must be only three entries to 60000119:0"
	exit 1
fi

make -f Makefile.nodebug clean
