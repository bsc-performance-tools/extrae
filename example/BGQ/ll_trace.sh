# @ job_name = trace
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

# Enable these if you don't have libxml2 support in Extrae
# export EXTRAE_ON=1
# export EXTRAE_MPI_COUNTERS_ON=1
# export EXTRAE_COUNTERS=PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_L1_LDM,PAPI_BR_MSP,PAPI_FP_INS,PAPI_TLB_DM

# Enable this if you have libxml2 support in Extrae
export EXTRAE_CONFIG_FILE=extrae.xml
runjob -n 8 --exe ./mpi_ping --env-all --ranks-per-node 8

