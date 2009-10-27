#!/bin/bash

export MPI_HOME=/usr/local/share/mpich-1.2.7p1

${MPI_HOME}/bin/mpirun -np 2 -machinefile hfile ./trace-xml.sh

