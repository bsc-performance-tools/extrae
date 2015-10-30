#!/bin/bash

rm -fr *.sym *.mpits set-0

EXTRAE_ON=1 ./define_event_type_gen_pcf_f
../../../src/merger/mpi2prv -without-addresses -f TRACE.mpits -e .libs/define_event_type_gen_pcf_f -o define_event_type_gen_pcf_f.prv

# Actual comparison
diff define_event_type_gen_pcf_f.reference define_event_type_gen_pcf_f.pcf
