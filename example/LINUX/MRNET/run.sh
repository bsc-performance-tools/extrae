#!/bin/bash
# @ initialdir = .
# @ output = job%j.out
# @ error =  job%j.err
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!! REQUEST AT LEAST 1 EXTRA TASK !!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# @ total_tasks = 33
# @ cpus_per_task = 1
# @ tasks_per_node = 4
# @ wall_clock_limit = 00:30:00

export EXTRAE_HOME=/home/bsc41/bsc41127/Tools/mpitrace_mrnet2.0/mpitrace_svn20100211/Package/64

### Set how many tasks will the application run with. 
### Note that you have to allocate extra resources above (at least 1 more task).
APPL_NAME=./test
APPL_NPROCS=32

### Run the application with: mrnrun <nprocs> ./trace.sh <appl>
time ${EXTRAE_HOME}/bin/mrnrun ${APPL_NPROCS} ./trace.sh ${APPL_NAME}
