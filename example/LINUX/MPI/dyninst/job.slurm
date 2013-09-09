#!/bin/sh
# @ initialdir = .
# @ output = trace.out
# @ error =  trace.err
# @ total_tasks = 4
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 00:10:00
# @ tracing = 1

srun ./trace.sh ./mpi_ping

