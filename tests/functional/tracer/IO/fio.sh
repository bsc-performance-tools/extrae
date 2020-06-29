#!/usr/bin/env sh

source ../../helper_functions.bash

TRACE=${0/\.sh/}

rm -fr ${TRACE}.{pcf,prv,row} set-0 TRACE.*

./trace.sh ./fio

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf fopen\(\)
CheckEntryInPCF ${TRACE}.pcf fwrite\(\)
CheckEntryInPCF ${TRACE}.pcf fread\(\)
CheckEntryInPCF ${TRACE}.pcf fclose\(\)

NumberEntriesInPRV ${TRACE}.prv 40000004 12
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one fopen() call"
 	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 4
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one fwrite() call"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 3
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one fread() call"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 15
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one fclose() call"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 40000004 0
if [[ "${?}" -ne 4 ]] ; then
	echo "There must be exactly 4 IO calls exits"
	exit 1
fi

exit 0
