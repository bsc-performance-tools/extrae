#!/bin/sh
# @ initialdir = .
# @ output = parallel_merge.out
# @ error =  parallel_merge.err
# @ total_tasks = 4
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 00:10:00

source @sub_PREFIXDIR@/etc/extrae.sh

srun ${EXTRAE_HOME}/bin/mpimpi2prv -syn -f TRACE.mpits -o trace.prv

