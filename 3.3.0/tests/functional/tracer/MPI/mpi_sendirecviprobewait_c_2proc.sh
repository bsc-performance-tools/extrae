#!/bin/bash

source ../../helper_functions.bash

if test -x ./pass_argument_MPIRUN ; then
	MPIRUN=`./pass_argument_MPIRUN`
else
	exit 1
fi

rm -fr TRACE.* *.mpits set-0

TRACE=${0/\.sh/}

EXTRAE_CONFIG_FILE=extrae.xml ${MPIRUN} -np 2 ./trace-ldpreload.sh ./mpi_sendirecviprobewait_c

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf MPI_Init
CheckEntryInPCF ${TRACE}.pcf MPI_Finalize
CheckEntryInPCF ${TRACE}.pcf MPI_Iprobe
CheckEntryInPCF ${TRACE}.pcf "MPI_Iprobe misses"
CheckEntryInPCF ${TRACE}.pcf "Elapsed time outside MPI_Iprobe"

NumberEntriesInPRV ${TRACE}.prv 50000003 31
if [[ "${?}" -ne 2 ]] ; then
	die "There must be only two entries to MPI_Init"
fi

NumberEntriesInPRV ${TRACE}.prv 50000003 32
if [[ "${?}" -ne 2 ]] ; then
	die "There must be only two entries to MPI_Finalize"
fi

NumberEntriesInPRV ${TRACE}.prv 50000001 62
if [[ "${?}" -ne 1 ]] ; then
	die "There must be only one entry to MPI_Iprobe"
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
