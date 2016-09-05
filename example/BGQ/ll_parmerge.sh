# @ job_name = parmerge
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

source @sub_PREFIXDIR@/etc/extrae.sh

runjob -n 2 : ${EXTRAE_HOME}/bin/mpimpi2prv -f TRACE.mpits
