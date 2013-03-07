#!/bin/sh

export EXTRAE_HOME=@sub_PREFIXDIR@

# Load the online environment
source ${EXTRAE_HOME}/etc/online_env.sh

# Select the Extrae configuration file
export EXTRAE_CONFIG_FILE=./extrae_online.xml

# Preload the tracing library 
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so    # C programs
#export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitracef.so  # Fortran programs

## Run the program
$*

