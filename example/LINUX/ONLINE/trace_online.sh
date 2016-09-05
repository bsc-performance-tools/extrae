#!/bin/sh

source @sub_PREFIXDIR@/etc/extrae.sh

# Load the online environment
source ${EXTRAE_HOME}/etc/extrae_online.sh

# Set the Extrae configuration
export EXTRAE_CONFIG_FILE=./extrae_online.xml
#export EXTRAE_ONLINE_DEBUG=1

# Start the analysis front-end 
if test "x${OMPI_COMM_WORLD_RANK}" = "x0" -o "x${SLURM_PROCID}" = "x0" -o "x${PMI_RANK}" = "x0" -o "x${MP_CHILD}" = "x0"; then
  ${EXTRAE_HOME}/bin/online_root &
fi 

# Preload the tracing library 
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so    # C programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so  # Fortran programs

# Run the program
$*

# Wait for the analysis front-end to quit gracefully
wait
