#!/bin/bash

# This script can be executed after installing the package
if test ! -r ../../../PREFIX ; then
	echo "Could not locate prefix installation"
	exit -1
fi

rm -fr TRACE.sym TRACE.mpits set-0

export EXTRAE_HOME=`cat ../../../PREFIX`

if test ! -x ${EXTRAE_HOME}/bin/extrae ; then
	echo "Could not execute binary rewriter test because ${EXTRAE_HOME}/bin/extrae has not been installed. Run make install first."
	exit 0
fi

source ${EXTRAE_HOME}/etc/extrae.sh

${EXTRAE_HOME}/bin/extrae -config extrae-nothing-to-instrument.xml ./pi

if test -f TRACE.sym ; then
	no_lines=`wc -l < TRACE.sym`
	if test "${no_lines}" == "0"; then
		rm TRACE.sym
		${EXTRAE_HOME}/bin/mpi2prv -dump-without-time -f TRACE.mpits -d >& MYDUMP
		grep ^TIME MYDUMP > MYDUMP2
		grep -v "EV: 40000033" MYDUMP2 > MYDUMP3
		rm -f MYDUMP MYDUMP2
		diff test-dyninst-2.reference MYDUMP3
	else
		echo "Could not generate a proper TRACE.sym (something written in there?"
		exit 2
	fi
else
	echo "Could not generate TRACE.sym"
	exit 1
fi

