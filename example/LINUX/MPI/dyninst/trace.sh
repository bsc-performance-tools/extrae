#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

# Only show output for task 0, others task send output to /dev/null
if test "${SLURM_PROCID}" == "0" -o \
        "${OMPI_COMM_WORLD_RANK}" == "0" -o \
        "${PMI_RANK}" == "0"; then
	${EXTRAE_HOME}/bin/extrae -config ../extrae.xml $@
else
	${EXTRAE_HOME}/bin/extrae -config ../extrae.xml $@ > /dev/null 2> /dev/null
fi

