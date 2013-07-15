#!/bin/sh

export EXTRAE_HOME=/home/harald/P-clock

EXTRAE_CONFIG_FILE=extrae.xml LD_PRELOAD=${EXTRAE_HOME}/lib/liboclmpitrace.so ./mpi-vadd
