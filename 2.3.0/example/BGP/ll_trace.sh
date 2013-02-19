# @ job_name = trace
# @ error = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL;
# @ wall_clock_limit = 00:10:00
# @ job_type = bluegene
# @ notify_user = user@address.com
# @ bg_size = 32
# @ queue

/usr/local/bin/mpirun -np 2 -exe `pwd`/mpi_ping -env "EXTRAE_ON=1 EXTRAE_MPI_COUNTERS_ON=1 EXTRAE_COUNTERS=PAPI_L2_DCM,PAPI_LST_INS,PNE_BGP_TORUS_XP_PACKETS,PNE_BGP_TORUS_XM_PACKETS,PNE_BGP_TORUS_YP_PACKETS,PNE_BGP_TORUS_YM_PACKETS,PNE_BGP_TORUS_ZP_PACKETS,PNE_BGP_TORUS_ZM_PACKETS"

