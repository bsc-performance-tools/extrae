#!/bin/sh

export MPITRACE_HOME=@sub_PREFIXDIR@
${MPITRACE_HOME}/bin/mpi2prv -e ./example TRACE*.mpit
