#!/bin/sh

export MPITRACE_HOME=/gpfs/apps/CEPBATOOLS/64.hwc
export MPTRACE_CONFIG_FILE=mpitrace.xml

export LD_PRELOAD=${MPITRACE_HOME}/lib/libmpitrace.so

## Run the desired program
$*

