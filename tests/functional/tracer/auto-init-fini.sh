#!/bin/bash

rm -fr *.sym *.mpits set-0

EXTRAE_ON=1 ./auto-init-fini
../../../src/merger/mpi2prv -f TRACE.mpits -dump -dump-without-time >& OUTPUT

# Remove headers for mpi2prv dump
grep -v ^mpi2prv OUTPUT   > OUTPUT-1
grep -v ^merger  OUTPUT-1 > OUTPUT-2
grep -v "EV: 40000033" OUTPUT-2 > OUTPUT-3
rm -f OUTPUT OUTPUT-1 OUTPUT-2

# Actual comparison
diff auto-init-fini.reference OUTPUT-3
