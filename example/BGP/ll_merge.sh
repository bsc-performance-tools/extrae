# @ job_name = merge
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

export MPITRACE_HOME=@sub_PREFIXDIR@

/usr/local/bin/mpirun -np 1 ${MPITRACE_HOME}/bin/mpi2prv -f TRACE.mpits

