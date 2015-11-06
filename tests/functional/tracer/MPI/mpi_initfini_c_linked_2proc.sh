#!/bin/bash

if test -x ./pass_argument_MPIRUN ; then
	MPIRUN=`./pass_argument_MPIRUN`
else
	exit 1
fi

rm -fr TRACE.* *.mpits set-0

TRACE=mpi_initfini_c_linked_2proc

EXTRAE_CONFIG_FILE=extrae.xml ${MPIRUN} -np 2 ./trace-static.sh ./mpi_initfini_c_linked

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
NENTRIES_INIT=`grep :50000003:31 ${TRACE}.prv | wc -l`
NENTRIES_FINI=`grep :50000003:32 ${TRACE}.prv | wc -l`
NEXITS=`grep :50000003:0 ${TRACE}.prv | wc -l`

NENTRIES_INIT_PCF=`grep MPI_Init ${TRACE}.pcf | wc -l`
NENTRIES_FINI_PCF=`grep MPI_Finalize ${TRACE}.pcf | wc -l`

if [[ "${NENTRIES_INIT_PCF}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Init in the PCF"
	exit 1
fi
if [[ "${NENTRIES_FINI_PCF}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Finalize in the PCF"
	exit 1
fi

if [[ "${NENTRIES_INIT}" -ne 2 ]] ; then
	echo "There must be only two entries to MPI_Init"
	exit 1
fi

if [[ "${NENTRIES_FINI}" -ne 2 ]] ; then
	echo "There must be only two entries to MPI_Finalize"
	exit 1
fi

if [[ "${NEXITS}" -ne 4 ]] ; then
	echo "There must be only four exits (one per MPI_Init and another per MPI_Finalize"
	exit 1
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
