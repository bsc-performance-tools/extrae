#!/bin/sh
# @ initialdir = .
# @ output = ping_%j.out
# @ error =  ping_%j.err
# @ total_tasks = 8
# @ cpus_per_task = 1
# @ wall_clock_limit = 00:10:00

APPL_NAME=./ping

srun ./trace_online.sh $APPL_NAME

