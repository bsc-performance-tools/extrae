#! /bin/tcsh
#@ job_type = parallel
#@ output = trace.ouput
#@ error = trace.error
#@ blocking = unlimited
#@ total_tasks = 2
#@ class = debug
#@ wall_clock_limit = 00:10:00
#@ restart = no
#@ group = bsc41 
#@ queue

setenv MLIST /tmp/machine_list.$$
/opt/ibmll/LoadL/full/bin/ll_get_machine_list > ${MLIST}
set NP = `cat ${MLIST} | wc -l`

mpirun -np ${NP} -machinefile ${MLIST} ./trace.sh ./mpi_ping

rm ${MLIST}
