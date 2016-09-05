#!/bin/bash
#BSUB -n 4
#BSUB -o merge_par_%J.out
#BSUB -e merge_par_%J.err
#BSUB -R"span[ptile=16]"
#BSUB -x 
#BSUB -J parallel
#BSUB -W 00:30

module load intel openmpi

source @sub_PREFIXDIR@/etc/extrae.sh

APPL_NAME=./ping
OUTPUT_TRACE=ping_online.prv

mpirun -np 4 ${EXTRAE_HOME}/bin/mpimpi2prv -syn -f TRACE.mpits -e $APPL_NAME -o $OUTPUT_TRACE

