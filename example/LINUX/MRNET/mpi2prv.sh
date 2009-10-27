#!/bin/bash

export MPITRACE_HOME=@sed_MYPREFIXDIR@

TRACE_NAME=trace_mpi_ping.prv

${MPITRACE_HOME}/bin/mpi2prv -f TRACE.mpits -o ${TRACE_NAME}

