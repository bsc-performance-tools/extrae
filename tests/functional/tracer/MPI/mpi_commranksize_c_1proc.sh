#!/bin/bash

source ../../helper_functions.bash

if test -x ./pass_argument_MPIRUN ; then
	MPIRUN=`./pass_argument_MPIRUN`
else
	exit 1
fi

rm -fr TRACE.* *.mpits set-0

TRACE=mpi_commsizerank_c

EXTRAE_CONFIG_FILE=extrae.xml ${MPIRUN} -np 1 ./trace-ldpreload.sh ./mpi_commranksize_c

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf MPI_Init

CheckEntryInPCF ${TRACE}.pcf MPI_Finalize

CheckEntryInPCF ${TRACE}.pcf MPI_Comm_rank

CheckEntryInPCF ${TRACE}.pcf MPI_Comm_size

NumberEntriesInPRV ${TRACE}.prv 50000003 31
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Init"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 50000003 32
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Finalize"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 50000003 19
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Comm_rank"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 50000003 20
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Comm_size"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 50000003 0
if [[ "${?}" -ne 4 ]] ; then
	echo "There must be only two exits (one per MPI_Init and another per MPI_Finalize)"
	exit 1
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
