#! /bin/ksh
#####################################################################
## Tracing step
#####################################################################
#@ step_name = trace_step
#@ job_type = parallel
#@ output = trace.out
#@ error = trace.err
#@ blocking = 2
#@ total_tasks = 4
#@ class = q09
#@ wall_clock_limit = 01:00:00
#@ restart = no
#@ group = bsc41
#@ node_usage = not_shared
#@ queue
#####################################################################
## Merging step
#####################################################################
#@ dependency = (trace_step == 0)
#@ step_name = merge_step
#@ job_type = serial
#@ output = merge.out
#@ error = merge.err
#@ wall_clock_limit = 01:00:00
#@ group = bsc41
#@ class = q09
#@ queue
#####################################################################

export EXTRAE_HOME=@sub_PREFIXDIR@

MLIST=~/machine_list.$$
/opt/ibmll/LoadL/full/bin/ll_get_machine_list > ${MLIST}
NP=`cat ${MLIST} | wc -l`

case ${LOADL_STEP_NAME} in
  trace_step)
    mpirun -np ${NP} -machinefile ${MLIST} ./trace.sh ./mpi_ping
    ;;
  merge_step)
    ${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -syn -o trace.prv
    ;;
  *)
    echo "Uknown step ${LOADL_STEP_NAME}"
    ;;
esac

