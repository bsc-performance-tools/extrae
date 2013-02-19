#!/bin/sh
# @ initialdir = .
# @ output = seq_merge.out
# @ error =  seq_merge.err
# @ total_tasks = 1
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 01:00:00

export EXTRAE_HOME=@sub_PREFIXDIR@

${EXTRAE_HOME}/bin/mpi2prv -syn -f TRACE.mpits -e ./mpi_ping -o trace.prv

