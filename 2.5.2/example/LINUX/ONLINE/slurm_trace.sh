#!/bin/bash
# @ initialdir = .
# @ output = trace.out
# @ error =  trace.err
# @ total_tasks = 4
# @ cpus_per_task = 1
# @ wall_clock_limit = 00:05:00

srun ./trace.sh ./test

