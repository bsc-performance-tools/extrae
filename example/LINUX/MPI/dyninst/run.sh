#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@
source ${EXTRAE_HOME}/etc/extrae.sh

# Only show output for task 0, others task send output to /dev/null
if test "${MXMPI_ID}" == "0" ; then
	${EXTRAE_HOME}/bin/ompitrace -config ../extrae.xml $@ > job${SLURM_JOB_ID}.${MXMPI_ID}.out 2> job${SLURM_JOB_ID}.${MXMPI_ID}.err
else
	${EXTRAE_HOME}/bin/ompitrace -config ../extrae.xml $@ > /dev/null 2> /dev/null
fi

