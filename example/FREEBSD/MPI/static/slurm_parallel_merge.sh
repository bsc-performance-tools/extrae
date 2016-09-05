#!/bin/sh
# @ initialdir = .
# @ output = par_merge.out
# @ error =  par_merge.err
# @ total_tasks = 5
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 01:00:00

source @sub_PREFIXDIR@/etc/extrae.sh

srun ${EXTRAE_HOME}/bin/mpimpi2prv -syn -f TRACE.mpits -o trace.prv

