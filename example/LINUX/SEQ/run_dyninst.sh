#!/bin/sh

export EXTRAE_HOME=/home/harald/T
export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib
source ${EXTRAE_HOME}/etc/extrae.sh

${EXTRAE_HOME}/bin/ompitrace -config extrae.xml ./pi

${EXTRAE_HOME}/bin/mpi2prv -e pi -f TRACE.mpits -s TRACE.sym -o pi.prv
