#!/bin/sh

rm -fr TRACE.sym TRACE.mpits set-0

EXTRAE_ON=1 ./check-ompss-codelocation
../../../src/merger/mpi2prv -f TRACE.mpits -dump -dump-without-time >& OUTPUT

# Remove headers for mpi2prv dump
grep -v ^mpi2prv OUTPUT   > OUTPUT-1
grep -v ^merger  OUTPUT-1 > OUTPUT-2

# Actual comparison
diff reference-2 OUTPUT-2
