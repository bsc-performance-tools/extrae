#!/bin/bash
#BSUB -n 8
#BSUB -o ping_%J.out
#BSUB -e ping_%J.err
#BSUB -R"span[ptile=16]"
#BSUB -x 
#BSUB -J parallel
#BSUB -W 00:10

module load intel openmpi

APPL_NAME=./ping

mpirun -np 8 ./trace_online.sh $APPL_NAME

