#!/bin/sh

export MPITRACE_HOME=../../..
${MPITRACE_HOME}/bin/mpi2prv -e ./example TRACE*.mpit
