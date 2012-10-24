#!/bin/bash

rm -fr TRACE.sym TRACE.mpits set-0

EXTRAE_ON=1 ./ompss-codelocation

# Actual comparison
diff ompss-codelocation.reference TRACE.sym
