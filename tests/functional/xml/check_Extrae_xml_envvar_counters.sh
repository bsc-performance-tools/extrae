#!/bin/bash

source ../helper_functions.bash
source ../../../etc/extrae.sh

rm -fr TRACE* set-0

TRACE=check_Extrae_xml_PAPI_TOT_INS

COUNTERS=PAPI_TOT_INS EXTRAE_CONFIG_FILE=extrae_envvar_counters.xml ./check_Extrae_xml
../../../src/merger/mpi2prv -f TRACE.mpits -o ${TRACE}.prv

# Check
if ! command -v papi_avail &> /dev/null
then
        echo "papi_avail could not be found"
        exit 0
fi

PAPI_TOT_CYC_available=`papi_avail | grep PAPI_TOT_CYC | awk '{print $3}'`
if [[ "$PAPI_TOT_CYC_available" == No ]]
then
        echo "PAPI_TOT_CYC is not available"
        exit 0
fi

CheckEntryInPCF ${TRACE}.pcf PAPI_TOT_INS

rm -fr TRACE* set-0 ${TRACE}.???

exit 0
