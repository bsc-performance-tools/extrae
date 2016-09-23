#!/bin/bash

source ../../helper_functions.bash

TRACE=JavaSimple

rm -fr TRACE.* *.mpits set-0

BUILD_DIR=../../../..
EXTRAEJ=${BUILD_DIR}/src/launcher/java/extraej.bash
export EXTRAEJ_LIBPTTRACE_PATH=${BUILD_DIR}/src/tracer/.libs/libpttrace.so
export EXTRAEJ_JAVATRACE_PATH=${BUILD_DIR}/src/java-connector/jni/javatrace.jar
export EXTRAEJ_LIBEXTRAEJVMTIAGENT_PATH=${BUILD_DIR}/src/java-connector/jvmti-agent/.libs/libextrae-jvmti-agent.so

EXTRAE_CONFIG_FILE=extrae.xml ${EXTRAEJ} -- JavaSimple

../../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

if [[ -r ${TRACE}.prv &&  -r ${TRACE}.pcf && -r ${TRACE}.row ]]; then
	rm -fr TRACE.* *.mpits set-0 ${TRACE}.pcf ${TRACE}.row ${TRACE}.prv
	exit 0
else
	die "Error checking existance for trace ${TRACE}*"
fi
