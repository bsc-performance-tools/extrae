#!/bin/sh

export OMP_NUM_THREADS=4
export MPTRACE_CONFIG_FILE=mpitrace.xml
export LD_LIBRARY_PATH=../../../lib

./pi
