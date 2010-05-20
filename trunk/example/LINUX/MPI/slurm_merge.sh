#!/bin/bash
# @ initialdir = .
# @ output = seq_merge.out
# @ error =  seq_merge.err
# @ total_tasks = 1
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 01:00:00

export MPITRACE_HOME=@sub_PREFIXDIR@

${MPITRACE_HOME}/bin/mpi2prv -syn -f TRACE.mpits -o trace.prv

