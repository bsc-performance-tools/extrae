#!/bin/sh
# @ initialdir = .
# @ output = merge.out
# @ error =  merge.err
# @ total_tasks = 1
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 00:10:00

source @sub_PREFIXDIR@/etc/extrae.sh

${EXTRAE_HOME}/bin/mpi2prv -syn -f TRACE.mpits -e ./mpi_ping -o trace.prv
