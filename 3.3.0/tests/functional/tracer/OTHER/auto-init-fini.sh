#!/bin/bash

source ../../helper_functions.bash

rm -fr *.sym *.mpits set-0

TRACE=${0/\.sh/}

EXTRAE_ON=1 ./auto-init-fini
../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison

NumberEntriesInPRV ${TRACE}.prv 1234 1
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :1234:1"
fi

NumberEntriesInPRV ${TRACE}.prv 1234 0
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :1234:0"
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
