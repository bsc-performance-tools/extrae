#!/bin/sh

###########################################################################
# Shared library preloading example (README)
#
# ** IMPORTANT ** Your binary must be compiled with a MPI shared library so
# as to run this example.
#
# So as to trace with the dynamic library, just set LD_PRELOAD to the .so
# Extrae library and set the required environment variables. In this
# example we set EXTRAE_CONFIG_FILE to configure the tracing.
#
# At the end of the script place the binary to be launched by mpirun.
#
# Finally, to launch this script use (for example with mpich):
#
#   mpirun -np ${NUM_PROCS} -machinefile ${MACHINE_FILE} ./run-dynamic.sh
#
###########################################################################

export LD_PRELOAD=@sub_PREFIXDIR@/lib/libmpitrace.so
export EXTRAE_CONFIG_FILE=extrae.xml

./mpi_ping

