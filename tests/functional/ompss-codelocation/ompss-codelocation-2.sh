#!/bin/sh

rm -fr TRACE.sym TRACE.mpits set-0 omps-codelocation-2.???

EXTRAE_ON=1 ./ompss-codelocation-2
../../../src/merger/mpi2prv -f TRACE.mpits -e .libs/ompss-codelocation-2 -o ompss-codelocation-2.prv

# Actual comparison
diff reference-2 ompss-codelocation-2.pcf
