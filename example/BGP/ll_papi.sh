# @ job_name = papi_info
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

export PAPI_HOME=@sub_PAPI_HOME@

/usr/local/bin/mpirun -np 1 ${PAPI_HOME}/bin/papi_avail > papi_bgp_avail.txt
/usr/local/bin/mpirun -np 1 ${PAPI_HOME}/bin/papi_native_avail > papi_bgp_native_avail.txt

