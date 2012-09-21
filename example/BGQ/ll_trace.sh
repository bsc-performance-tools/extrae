# @ job_name = trace
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

export EXTRAE_ON=1
runjob -n 8 --exe ./mpi_ping --env-all --ranks-per-node 8

