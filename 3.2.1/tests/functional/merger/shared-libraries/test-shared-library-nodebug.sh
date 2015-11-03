#!/bin/bash

make -f Makefile.nodebug run

#tail -25 main.pcf > OUTPUT

make -f Makefile.nodebug clean

# Do test
diff test-shared-library-nodebug.reference main.pcf
