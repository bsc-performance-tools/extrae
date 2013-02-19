#!/bin/bash

export MPI_HOME=@sub_MPI_HOME@

${MPI_HOME}/bin/mpirun -np 2 -machinefile hfile ./trace-xml.sh

