#!/bin/bash

export EXTRAE_HOME=@sub_PREFIXDIR@

${EXTRAE_HOME}/bin/mpitrace /bin/dmpirun -np 2 ./mpi_ping
