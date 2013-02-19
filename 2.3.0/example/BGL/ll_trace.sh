#@ job_name = trace_step
#@ chips = 2
#@ wall_clock_limit = 00:10:00
#@ account_no = b12-upc
#@ tasks = 2
#@ arguments = -exe @sub_PREFIXDIR@/share/example/mpi_ping \
         -cwd @sub_PREFIXDIR@/share/example \
         -mode VN \
         -env "EXTRAE_ON=1 EXTRAE_COUNTERS=PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_L3_TCM,BGL_UPC_TS_XM_32B_CHUNKS,BGL_UPC_TS_XP_32B_CHUNKS,BGL_UPC_TS_YM_32B_CHUNKS,BGL_UPC_TS_YP_32B_CHUNKS,BGL_UPC_TS_ZM_32B_CHUNKS,BGL_UPC_TS_ZP_32B_CHUNKS EXTRAE_BUFFER_SIZE=100000 EXTRAE_MPI_COUNTERS_ON=1" \
         -np $(tasks)
#
#@ output = $(job_name).$(schedd_host).$(jobid).out
#@ error  = $(job_name).$(schedd_host).$(jobid).err
#
#Do not alter below unless you know what you are doing
#@ bg_size = $(chips)
#@ bg_connection = PREFER_TORUS
#@ executable = /bgl/BlueLight/ppcfloor/bglsys/bin/mpirun
#@ job_type = BLUEGENE
#@ environment = $MMCS_SERVER_IP;$BACKEND_MPIRUN_PATH
#@ class = BGL
#@ queue
