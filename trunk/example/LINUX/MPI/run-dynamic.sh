#!/bin/sh

###########################################################################
# Shared library preloading example (README)
#
# ** IMPORTANT ** Your binary must be compiled with a MPI shared library so
# as to run this example.
#
# So as to trace with the dynamic library, just set LD_PRELOAD to the .so
# MPItrace library and set the required environment variables. In this
# example we set MPITRACE_ON to activate the tracing and the MPTRACE_COUNTERS
# to select which counters we wanna obtain from PAPI.
#
# At the end of the script place the binary to be launched by mpirun.
#
# Finally, to launch this script use (for example with mpich):
#
#   mpirun -np ${NUM_PROCS} -machinefile ${MACHINE_FILE} ./run-dynamic.sh
#
###########################################################################

export LD_PRELOAD=${MPITRACE_HOME}/lib/libmpitrace.so
export MPITRACE_ON=1
export MPTRACE_COUNTERS=0x80000007,0x80000032
export MPITRACE_MPI_COUNTERS_ON=1
export MPTRACE_COUNTERS_DOMAIN=all
export MPITRACE_MPI_CALLER=1,2

./mpi_ping

