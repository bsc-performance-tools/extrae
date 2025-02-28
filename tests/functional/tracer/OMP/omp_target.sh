#!/usr/bin/env sh

source ../../helper_functions.bash

TRACE=omp_target

rm -fr ${TRACE}.* *.mpits set-0

OMP_NUM_THREADS=4 EXTRAE_CONFIG_FILE=extrae.xml ./trace.sh ./omp_target

CheckEntryInPCF ${TRACE}.pcf "OpenMP target"

NumberEntriesInPRV ${TRACE}.prv 60000034 1
if [[ "${?}" -ne 3 ]] ; then
	die "There must be exactly three entries to OpenMP target"
fi

exit 0
