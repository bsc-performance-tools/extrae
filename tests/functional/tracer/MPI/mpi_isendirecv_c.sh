#!/bin/bash

source ../../helper_functions.bash

if test -x ./pass_argument_MPIRUN ; then
	MPIRUN=`./pass_argument_MPIRUN`
else
	exit 1
fi

rm -fr TRACE.* *.mpits set-0

TRACE=${0/\.sh/}

EXTRAE_CONFIG_FILE=extrae.xml ${MPIRUN} -np 1 ./trace-ldpreload.sh ./mpi_isendirecv_c

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf MPI_Init
CheckEntryInPCF ${TRACE}.pcf MPI_Isend
CheckEntryInPCF ${TRACE}.pcf MPI_Irecv
CheckEntryInPCF ${TRACE}.pcf MPI_Wait
CheckEntryInPCF ${TRACE}.pcf MPI_Finalize

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

NumberEntriesInPRV ${TRACE}.prv 50000001 5
if [[ "${?}" -ne 2 ]] ; then
	echo "There must be only one entry to MPI_Wait"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 50000001 4
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Irecv"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 50000001 3
if [[ "${?}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Isend"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 50000001 0
if [[ "${?}" -ne 4 ]] ; then
	echo "There must be four MPI p2p exits"
	exit 1
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
