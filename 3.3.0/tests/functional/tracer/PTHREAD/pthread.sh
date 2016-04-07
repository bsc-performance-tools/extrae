#!/bin/bash

source ../../helper_functions.bash

rm -fr TRACE.* *.mpits set-0

TRACE=pthread

EXTRAE_CONFIG_FILE=extrae.xml ./trace-ldpreload.sh ./pthread

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

CheckEntryInPCF ${TRACE}.pcf pthread_create
CheckEntryInPCF ${TRACE}.pcf pthread_join
CheckEntryInPCF ${TRACE}.pcf pthread_func

NB_PTHREAD_FUNC_EVTS=`grep :60000020: ${TRACE}.prv | wc -l`

if [[ "${NB_PTHREAD_FUNC_EVTS}" -ne 4 ]]; then
	die "Number of pthread_func events in PCF should be 4 (2 entries, 2 exits)"
fi

rm -fr TRACE.* set-0 pthread.prv pthread.pcf pthread.row

exit 0
