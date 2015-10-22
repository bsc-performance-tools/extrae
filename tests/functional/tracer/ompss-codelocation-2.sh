#!/bin/bash

rm -fr *.sym *.mpits set-0

EXTRAE_ON=1 ./ompss-codelocation-2
../../../src/merger/mpi2prv -without-addresses -f TRACE.mpits -e .libs/ompss-codelocation-2 -o ompss-codelocation-2.prv

# Actual comparison
diff ompss-codelocation-2.reference ompss-codelocation-2.pcf
