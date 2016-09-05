# @ job_name = merge
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

source @sub_PREFIXDIR@/etc/extrae.sh

runjob -n 1 : ${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits
