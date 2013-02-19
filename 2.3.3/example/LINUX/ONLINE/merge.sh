#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@

${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -e ./test -o online_test_trace.prv

