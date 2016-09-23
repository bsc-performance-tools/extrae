#!/bin/bash

source ../../helper_functions.bash

BUILD_DIR=../../../..
EXTRAEJ=${BUILD_DIR}/src/launcher/java/extraej.bash
export EXTRAEJ_LIBPTTRACE_PATH=${BUILD_DIR}/src/tracer/.libs/libpttrace.so
export EXTRAEJ_JAVATRACE_PATH=${BUILD_DIR}/src/java-connector/jni/javatrace.jar
export EXTRAEJ_LIBEXTRAEJVMTIAGENT_PATH=${BUILD_DIR}/src/java-connector/jvmti-agent/.libs/libextrae-jvmti-agent.so

TRACE0=JavaThreads0
rm -fr TRACE.* *.mpits set-0
EXTRAE_CONFIG_FILE=extrae.xml ${EXTRAEJ} -- JavaThreads 0
../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE0}.prv

TRACE1=JavaThreads1
rm -fr TRACE.* *.mpits set-0
EXTRAE_CONFIG_FILE=extrae.xml ${EXTRAEJ} -- JavaThreads 1
../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE1}.prv

TRACE2=JavaThreads2
rm -fr TRACE.* *.mpits set-0
EXTRAE_CONFIG_FILE=extrae.xml ${EXTRAEJ} -- JavaThreads 2
../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE2}.prv

TRACE4=JavaThreads4
rm -fr TRACE.* *.mpits set-0
EXTRAE_CONFIG_FILE=extrae.xml ${EXTRAEJ} -- JavaThreads 4
../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE4}.prv

# Actual checks
CheckEntryInPCF ${TRACE1}.pcf "pthread_create"
CheckEntryInPCF ${TRACE2}.pcf "pthread_create"
CheckEntryInPCF ${TRACE4}.pcf "pthread_create"
CheckEntryInPCF ${TRACE1}.pcf "pthread_join"
CheckEntryInPCF ${TRACE2}.pcf "pthread_join"
CheckEntryInPCF ${TRACE4}.pcf "pthread_join"

NumberEntriesInPRV ${TRACE0}.prv 61000000 2
NTHREADS0=${?}

NumberEntriesInPRV ${TRACE1}.prv 61000000 2
NTHREADS1=${?}
if [[ "${NTHREADS1}" -lt 1 ]] ; then
	die "There must be at least one entry to pthread_create (1) - ${NTHREADS1}"
fi

NumberEntriesInPRV ${TRACE2}.prv 61000000 2
NTHREADS2=${?}
if [[ "${NTHREADS2}" -lt 2 ]] ; then
	die "There must be at least one entry to pthread_create (2) - ${NTHREADS2}"
fi

NumberEntriesInPRV ${TRACE4}.prv 61000000 2
NTHREADS4=${?}
if [[ "${NTHREADS4}" -lt 4 ]] ; then
	die "There must be at least one entry to pthread_create (4) - ${NTHREADS4}"
fi

if [[ ${NTHREADS0} -ne  $((NTHREADS1-1)) ]]; then
	die "NTHREADS0 ($NTHREADS0) should be one less than NTHREADS1 ($NTHREADS1)"
fi

if [[ ${NTHREADS0} -ne  $((NTHREADS2-2)) ]]; then
	die "NTHREADS0 ($NTHREADS0) should be one less than NTHREADS2 ($NTHREADS2)"
fi

if [[ ${NTHREADS0} -ne  $((NTHREADS4-4)) ]]; then
	die "NTHREADS0 ($NTHREADS0) should be one less than NTHREADS4 ($NTHREADS4)"
fi

rm -fr TRACE.* *.mpits set-0 *[0124].pcf *[0124].row *[0124].prv
exit 0
