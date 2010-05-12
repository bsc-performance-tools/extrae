#! /bin/ksh
#####################################################################
## Tracing step
#####################################################################
#@ step_name = trace_step
#@ job_type = parallel
#@ output = trace.out
#@ error = trace.err
#@ blocking = unlimited
#@ total_tasks = 4
#@ node_usage = not_shared
#@ class = q09
#@ wall_clock_limit = 01:00:00
#@ restart = no
#@ group = bsc41
#@ queue
#####################################################################
## Merging step
#####################################################################
#@ dependency = (trace_step == 0)
#@ step_name = merge_step
#@ job_type = parallel
#@ output = merge.out
#@ error = merge.err
#@ blocking = unlimited
#@ total_tasks = 3
#@ wall_clock_limit = 01:00:00
#@ group = bsc41
#@ class = q09
#@ queue
#####################################################################

MPITRACE_HOME=@sub_PREFIXDIR@
MLIST=~/machine_list.$$.${LOADL_STEP_NAME}
/opt/ibmll/LoadL/full/bin/ll_get_machine_list > ${MLIST}
NP=`cat ${MLIST} | wc -l`

case ${LOADL_STEP_NAME} in
	trace_step)
    mpirun -np ${NP} -machinefile ${MLIST} ./trace.sh ./mpi_ping
    ;;
	merge_step)
    mpirun -np ${NP} -machinefile ${MLIST} ${MPITRACE_HOME}/bin/mpimpi2prv -f TRACE.mpits -maxmem 1024 -syn -o trace.prv
    ;;
	*)
    echo "Uknown step ${LOADL_STEP_NAME}"
    ;;
esac

