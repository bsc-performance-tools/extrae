#!/usr/bin/env sh

source ../../helper_functions.bash

TRACE=${0/\.sh/}

rm -fr ${TRACE}.{pcf,prv,row} set-0 TRACE.*

./trace.sh ./io

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf open\(\)
CheckEntryInPCF ${TRACE}.pcf write\(\)
CheckEntryInPCF ${TRACE}.pcf read\(\)
CheckEntryInPCF ${TRACE}.pcf close\(\)

NumberEntriesInPRV ${TRACE}.prv 40000004 11
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one open() call"
 	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 2
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one write() call"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 1
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one read() call"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 14
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one close() call"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 0
if [[ "${?}" -ne 4 ]] ; then
	echo "There must be exactly 4 IO calls exits"
	exit 1
fi

exit 0
