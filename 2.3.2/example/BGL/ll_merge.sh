#@ job_name = merge_step
#@ chips = 1
#@ wall_clock_limit = 00:10:00
#@ account_no = b12-upc
#@ tasks = 1
#@ arguments = -exe @sub_PREFIXDIR@/bin/mpi2prv \
         -cwd @sub_PREFIXDIR@/share/example \
         -mode VN \
         -np $(tasks) \
         -args "-f TRACE.mpits -syn"
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



