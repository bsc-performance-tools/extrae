#!/bin/bash

source ../../helper_functions.bash

if test -x ./pass_argument_MPIRUN ; then
	MPIRUN=`./pass_argument_MPIRUN`
else
	exit 1
fi

rm -fr TRACE.* *.mpits set-0

TRACE=${0/\.sh/}

EXTRAE_CONFIG_FILE=extrae.xml ${MPIRUN} -np 4 ./trace-static.sh ./mpi_initfini_c_linked

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf MPI_Init
CheckEntryInPCF ${TRACE}.pcf MPI_Finalize

NumberEntriesInPRV ${TRACE}.prv 50000003 31
if [[ "${?}" -ne 4 ]] ; then
	die "There must be only four entries to MPI_Init"
fi

NumberEntriesInPRV ${TRACE}.prv 50000003 32
if [[ "${?}" -ne 4 ]] ; then
	die "There must be only four entries to MPI_Finalize"
fi

NumberEntriesInPRV ${TRACE}.prv 50000003 0
if [[ "${?}" -ne 8 ]] ; then
	die "There must be only eight exits (one per MPI_Init and another per MPI_Finalize"
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
