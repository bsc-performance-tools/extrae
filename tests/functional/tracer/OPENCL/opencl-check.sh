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

NEVENTSINACCEL=`grep ^2:2:1:1:2: opencl-check.prv | wc -l`
# 11 are:
#  1 counter set,  2 * 2 reads, 1 * 2 * ndrange, 1 * 2 * write, 1 * 2 flush
if test ${NEVENTSINACCEL} -ne 11 ; then
	echo "Number of events in accelerator should be 11, but it is ${NEVENTSINACCEL}
	exit 1
fi

NSTATESINACCEL=`grep ^1:2:1:1:2 opencl-check.prv | wc -l`
# 6 are:
#  1 not created, 3 memory transfer, 1 running, 1 flushing
if test ${NSTATESINACCEL} -ne 6 ; then
	echo "Number of events in accelerator should be 6, but it is ${NEVENTSINACCEL}
	exit 1
fi

diff opencl-check.pcf.reference opencl-check.pcf
