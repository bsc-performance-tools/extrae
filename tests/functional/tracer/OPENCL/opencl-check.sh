#!/bin/bash

rm -fr *.sym *.mpits set-0

EXTRAE_CONFIG_FILE=extrae.xml ./opencl-check

# Actual comparison
NENTRIES=`grep :64200000:1 opencl-check.prv | wc -l`
NEXITS=`grep :64200000:0 opencl-check.prv | wc -l`

if test ${NENTRIES} -ne 2 -o ${NENTRIES} -ne 2 ; then
	echo "Number of entries and exits must be equal to 2"
	exit 1
fi

diff opencl-check.reference opencl-check.pcf 
