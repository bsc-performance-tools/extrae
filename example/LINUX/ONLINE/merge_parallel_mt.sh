#!/bin/sh
# @ initialdir = .
# @ output = merge_par_%j.out
# @ error =  merge_par_%j.err
# @ total_tasks = 4
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 00:30:00

source @sub_PREFIXDIR@/etc/extrae.sh

APPL_NAME=./ping
OUTPUT_TRACE=ping_online.prv

srun ${EXTRAE_HOME}/bin/mpimpi2prv -syn -f TRACE.mpits -e $APPL_NAME -o $OUTPUT_TRACE

