#!/bin/bash

source ../../helper_functions.bash

rm -fr *.sym *.mpits set-0

TRACE=${0/\.sh/}

EXTRAE_ON=1 ./ompss-codelocation
../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf "pi_kernel"
CheckEntryInPCF ${TRACE}.pcf "sleep_kernel"
CheckEntryInPCF ${TRACE}.pcf "my_function"
CheckEntryInPCF ${TRACE}.pcf "fake_kernel"
CheckEntryInPCF ${TRACE}.pcf "37 (ompss-codelocation.c"
CheckEntryInPCF ${TRACE}.pcf "46 (ompss-codelocation.c"
CheckEntryInPCF ${TRACE}.pcf "69 (ompss-codelocation.c"
CheckEntryInPCF ${TRACE}.pcf "74 (ompss-codelocation.c"
CheckEntryInPCF ${TRACE}.pcf "91 (ompss-codelocation.c"

NumberEntriesInPRV ${TRACE}.prv 2000 3
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2000:3"
fi
NumberEntriesInPRV ${TRACE}.prv 2000 4
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2000:4"
fi
NumberEntriesInPRV ${TRACE}.prv 2000 6
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2000:6"
fi
NumberEntriesInPRV ${TRACE}.prv 2000 0
if [[ "${?}" -ne 3 ]] ; then
	die "There must be only one :2000:0"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 4
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2020:4"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 6
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2020:6"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 8
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one :2020:8"
fi
NumberEntriesInPRV ${TRACE}.prv 2020 0
if [[ "${?}" -ne 3 ]] ; then
	die "There must be only one :2020:0"
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
