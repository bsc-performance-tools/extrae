#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

if test x`uname -s` == xDarwin ; then
	EXTRAE_CONFIG_FILE=extrae.xml DYLD_FORCE_FLAT_NAMESPACE=1 DYLD_INSERT_LIBRARIES=${EXTRAE_HOME}/lib/libocltrace.dylib $@
else
	EXTRAE_CONFIG_FILE=extrae.xml LD_PRELOAD=${EXTRAE_HOME}/lib/libocltrace.so $@
fi
