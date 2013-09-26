#!/bin/sh
# @ initialdir = .
# @ output = trace.out
# @ error =  trace.err
# @ total_tasks = 2
# @ cpus_per_task = 1
# @ tasks_per_node = 2
# @ wall_clock_limit = 00:10:00
# @ gpus_per_node = 1
# @ tracing = 1

srun ./trace.sh ./mpi_hello

