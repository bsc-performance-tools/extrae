#!/bin/bash

source ../../helper_functions.bash

rm -fr *.sym *.mpits set-0

TRACE=${0/\.sh/}

EXTRAE_ON=1 ./define_event_type_gen_pcf
../../../../src/merger/mpi2prv -f TRACE.mpits -e .libs/define_event_type_gen_pcf -o ${TRACE}.prv

# Actual comparison
CheckEntryInPCF ${TRACE}.pcf "Kernel execution"
CheckEntryInPCF ${TRACE}.pcf "Kernel execution_2"
CheckEntryInPCF ${TRACE}.pcf "Phase1"
CheckEntryInPCF ${TRACE}.pcf "Phase2"

rm -fr ${TRACE}.prv ${TRACE}.pcf ${TRACE}.row set-0 TRACE.*
