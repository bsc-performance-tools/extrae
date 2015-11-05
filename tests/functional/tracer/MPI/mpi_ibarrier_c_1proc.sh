#!/bin/bash

if test -x ./pass_argument_MPIRUN ; then
	MPIRUN=`./pass_argument_MPIRUN`
else
	exit 1
fi

rm -fr TRACE.* *.mpits set-0

TRACE=mpi_initfini_c_linked_1proc

EXTRAE_CONFIG_FILE=extrae.xml ${MPIRUN} -np 1 ./trace-ldpreload.sh ./mpi_ibarrier_c

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual comparison
NENTRIES_INIT=`grep :50000003:31 ${TRACE}.prv | wc -l`
NENTRIES_FINI=`grep :50000003:32 ${TRACE}.prv | wc -l`
NENTRIES_IBARRIER=`grep :50000002:156 ${TRACE}.prv | wc -l`
NENTRIES_WAIT=`grep :50000002:156 ${TRACE}.prv | wc -l`
NEXITS=`grep :50000003:0 ${TRACE}.prv | wc -l`
NEXITS2=`grep :50000002:0 ${TRACE}.prv | wc -l`
NEXITS1=`grep :50000001:0 ${TRACE}.prv | wc -l`

if [[ "${NENTRIES_INIT}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Init"
	exit 1
fi

if [[ "${NENTRIES_FINI}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Finalize"
	exit 1
fi

if [[ "${NENTRIES_IBARRIER}" -ne 1 ]] ; then
	echo "There must be only one entry to MPI_Ibarrier"
	exit 1
fi

# if [[ "${NEXITS}" -ne 2  ]] ; then
# 	echo "There must be only two exits on others"
# 	exit 1
# fi
if [[ "${NEXITS1}" -ne 1  ]] ; then
	echo "There must be only one exit on p2p"
	exit 1
fi
if [[ "${NEXITS2}" -ne 1  ]] ; then
 	echo "There must be only one exit on collective"
 	exit 1
fi

rm -fr ${TRACE}.??? set-0 TRACE.*

exit 0
