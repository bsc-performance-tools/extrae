#!/bin/bash

rm -fr TRACE.* *.mpits set-0

TRACE=pthread

EXTRAE_CONFIG_FILE=extrae.xml ./trace-ldpreload.sh ./pthread

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

NB_PTHREAD_CREATE=`grep pthread_create ${TRACE}.pcf | wc -l`
NB_PTHREAD_JOIN=`grep pthread_join ${TRACE}.pcf | wc -l`
NB_PTHREAD_FUNC=`grep pthread_func ${TRACE}.pcf | wc -l`
NB_PTHREAD_FUNC_EVTS=`grep :60000020: ${TRACE}.prv | wc -l`

if [[ "${NB_PTHREAD_CREATE}" -ne 1 ]]; then
	die "Number of pthread_create in PCF should be 1"
fi

if [[ "${NB_PTHREAD_JOIN}" -ne 1 ]]; then
	die "Number of pthread_join in PCF should be 1"
fi

if [[ "${NB_PTHREAD_FUNC}" -ne 1 ]]; then
	die "Number of pthread_func in PCF should be 1"
fi

if [[ "${NB_PTHREAD_FUNC_EVTS}" -ne 4 ]]; then
	die "Number of pthread_func events in PCF should be 4 (2 entries, 2 exits)"
fi

rm -fr TRACE.* set-0 pthread.prv pthread.pcf pthread.row

exit 0
