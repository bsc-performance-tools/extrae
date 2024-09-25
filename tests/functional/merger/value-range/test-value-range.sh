#!/bin/bash

source ../../helper_functions.bash

TRACE=main

EXTRAE_ON=1 ./main
../../../../src/merger/mpi2prv -f TRACE.mpits

# Do actual checks
CheckEntryInPCF ${TRACE}.pcf "18446744073709551614"

