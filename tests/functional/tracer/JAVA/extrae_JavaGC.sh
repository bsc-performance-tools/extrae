#!/bin/bash

source ../../helper_functions.bash

TRACE=JavaGC

rm -fr TRACE.* *.mpits set-0

BUILD_DIR=../../../..
EXTRAEJ=${BUILD_DIR}/src/launcher/java/extraej.bash
export EXTRAEJ_LIBPTTRACE_PATH=${BUILD_DIR}/src/tracer/.libs/libpttrace.so
export EXTRAEJ_JAVATRACE_PATH=${BUILD_DIR}/src/java-connector/jni/javatrace.jar
export EXTRAEJ_LIBEXTRAEJVMTIAGENT_PATH=${BUILD_DIR}/src/java-connector/jvmti-agent/.libs/libextrae-jvmti-agent.so

EXTRAE_CONFIG_FILE=extrae.xml ${EXTRAEJ} -- JavaGC

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Actual checks
CheckEntryInPCF ${TRACE}.pcf "Java Garbage collector"

NumberEntriesInPRV ${TRACE}.prv 48000001 1
NENTRIES=${?}
if [[ "${NENTRIES}" -lt 1 ]] ; then
	echo "There must be at least one entry to Java Garbage collector"
	exit 1
fi

NumberEntriesInPRV ${TRACE}.prv 48000001 0
NEXITS=${?}
if [[ "${NEXITS}" -ne "${NENTRIES}" ]] ; then
	die "There must be the same number of entries to exits in Java Garbage collector (${NEXITS} / ${NENTRIES})"
fi

if [[ -r ${TRACE}.prv &&  -r ${TRACE}.pcf && -r ${TRACE}.row ]]; then
	rm -fr TRACE.* *.mpits set-0 ${TRACE}.pcf ${TRACE}.row ${TRACE}.prv
	exit 0
else
	die "Error checking existance for trace ${TRACE}*"
fi
