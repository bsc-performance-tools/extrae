#!/bin/bash

make -f Makefile.debug run

tail -30 EXTRAE_Paraver_trace.pcf > OUTPUT

make -f Makefile.debug clean

# Do test
diff test-shared-library-debug.reference OUTPUT
