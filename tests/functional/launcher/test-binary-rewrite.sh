#!/bin/bash

# This script can be executed after installing the package
if test ! -r ../../../PREFIX ; then
	echo "Could not locate prefix installation"
	exit -1
fi

rm -fr TRACE.sym TRACE.mpits set-0

export EXTRAE_HOME=`cat ../../../PREFIX`
source ${EXTRAE_HOME}/etc/extrae.sh

${EXTRAE_HOME}/bin/extrae -config extrae.xml -rewrite ./pi

if test -x pi.extrae ; then
	rm -f pi.extrae
	no_lines=`grep ^U TRACE.sym | grep pi_kernel | wc -l`
	if test "${no_lines}" == "1" ; then
		exit 0
	else
		echo "Could not generate a proper TRACE.sym (did not find pi_kernel)"
		exit 2
	fi
else
	echo "Could not generate TRACE.sym"
	exit 1
fi

