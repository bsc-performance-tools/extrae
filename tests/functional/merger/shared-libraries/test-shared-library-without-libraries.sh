#!/bin/bash

make -f Makefile.debug run-nomerge
rm -fr set-0/*.sym
../../../../src/merger/mpi2prv -without-addresses -f TRACE.mpits -e ./main -o main.prv

# tail -25 EXTRAE_Paraver_trace.pcf > OUTPUT

make -f Makefile.debug clean

# Do test
diff test-shared-library-without-libraries.reference main.pcf
