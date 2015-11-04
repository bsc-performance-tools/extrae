#!/bin/bash

rm -fr TRACE.* *.mpits set-0

TRACE=mpi_initfini_c_linked.prv

EXTRAE_CONFIG_FILE=extrae.xml ./mpi_initfini_c_linked

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}

rm -fr TRACE.* *.mpits set-0

# Actual comparison
NENTRIES_INIT=`grep :50000003:31 ${TRACE} | wc -l`
NENTRIES_FINI=`grep :50000003:32 ${TRACE} | wc -l`
NEXITS=`grep :50000003:0 ${TRACE} | wc -l`

if [[ "${NENTRIES_INIT}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Init"
	exit
fi

if [[ "${NENTRIES_FINI}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Finalize"
	exit
fi

if [[ "${NEXITS}" -ne 2 ]] ; then
	echo "There must be only two exits (one per MPI_Init and another per MPI_Finalize"
	exit
fi
