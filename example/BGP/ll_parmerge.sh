# @ job_name = parmerge
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

EXTRAE_HOME=@sub_PREFIXDIR@
/usr/local/bin/mpirun -np 2 -exe ${EXTRAE_HOME}/bin/mpimpi2prv -args "-f TRACE.mpits"

