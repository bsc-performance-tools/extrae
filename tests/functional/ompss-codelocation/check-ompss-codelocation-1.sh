#!/bin/sh

rm -fr TRACE.sym TRACE.mpits set-0

EXTRAE_ON=1 ./check-ompss-codelocation

# Actual comparison
diff reference-1 TRACE.sym
