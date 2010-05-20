#!/bin/bash

export MPITRACE_HOME=@sub_PREFIXDIR@
${MPITRACE_HOME}/bin/mpi2prv *.mpit -syn
