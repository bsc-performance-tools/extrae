#!/bin/bash

source ../../helper_functions.bash

rm -fr *.sym *.mpits set-0

TRACE=${0/\.sh/}

EXTRAE_ON=1 ./ompss-codelocation-2
../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf "pi_kernel"
CheckEntryInPCF ${TRACE}.pcf "sleep_kernel"
CheckEntryInPCF ${TRACE}.pcf "fake_kernel"
CheckEntryInPCF ${TRACE}.pcf "39.*ompss-codelocation-2.c"
CheckEntryInPCF ${TRACE}.pcf "67.*ompss-codelocation-2.c"
CheckEntryInPCF ${TRACE}.pcf "91.*ompss-codelocation-2.c"
CheckEntryInPCF ${TRACE}.pcf "113.*ompss-codelocation-2.c"

NumberEntriesInPRV ${TRACE}.prv 2000 3
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2000:3"
fi
NumberEntriesInPRV ${TRACE}.prv 2000 4
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2000:4"
fi
NumberEntriesInPRV ${TRACE}.prv 2000 5
if [[ "${?}" -ne 2 ]] ; then
	die "There must be only one :2000:5"
fi
NumberEntriesInPRV ${TRACE}.prv 2000 0
if [[ "${?}" -ne 4 ]] ; then
	die "There must be only one :2020:0"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 3
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2020:3"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 4
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2020:4"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 5
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2020:5"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 6
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2020:6"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 0
if [[ "${?}" -ne 4 ]] ; then
	die "There must be only one :2020:0"
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
