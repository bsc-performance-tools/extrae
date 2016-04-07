#!/bin/bash

source ../helper_functions.bash

rm -fr TRACE* set-0

TRACE=check_Extrae_counters_xml_INS_CYC

EXTRAE_CONFIG_FILE=extrae-PAPI_TOT_INS_CYC.xml ./check_Extrae_counters_xml
../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Check
CheckEntryInPCF ${TRACE}.pcf PAPI_TOT_INS
CheckEntryInPCF ${TRACE}.pcf PAPI_TOT_CYC

rm -fr TRACE* set-0 ${TRACE}.???

exit 0
