# @ shell = /bin/ksh 
# @ job_name = ping
# @ job_type = parallel
# @ cpus = 2
# @ node_usage = not_shared
# @ output = $(job_name).$(cluster).$(process).out
# @ error = $(job_name).$(cluster).$(process).err
# @ wall_clock_limit = 00:30:00 
# @ account_no = z06-pavr
# @ notification = never
# @ initialdir = .
# @ total_tasks = 2
# @ queue

export EXTRAE_CONFIG_FILE=extrae.xml
poe ./mpi_ping

