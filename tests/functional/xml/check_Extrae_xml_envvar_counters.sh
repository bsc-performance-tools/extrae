#!/bin/bash

source ../helper_functions.bash
source ../../../etc/extrae.sh

rm -fr TRACE* set-0

TRACE=check_Extrae_xml_PAPI_TOT_INS

COUNTERS=PAPI_TOT_INS EXTRAE_CONFIG_FILE=extrae_envvar_counters.xml ./check_Extrae_xml
../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Check
CheckEntryInPCF ${TRACE}.pcf PAPI_TOT_INS

rm -fr TRACE* set-0 ${TRACE}.???

exit 0
