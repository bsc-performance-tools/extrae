#!/bin/bash
# @ class = debug
# @ initialdir = .
# @ output = trace%j.out
# @ error =  trace%j.err
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!! REQUEST AT LEAST 4 MORE TASKS !!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# @ total_tasks = 8
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 00:10:00

export MPITRACE_HOME=@sed_MYPREFIXDIR@

# Set how many tasks will the application run with
APPL_NPROCS=4
APPL_NAME=./mpi_ping

${MPITRACE_HOME}/bin/mrnrun ${APPL_NPROCS} ./trace.sh ${APPL_NAME}
