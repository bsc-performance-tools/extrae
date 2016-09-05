#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o ping_online.prv
