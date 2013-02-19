#!/bin/sh

export EXTRAE_HOME=/home/bsc41/bsc41127/Tools/mpitrace_mrnet2.0/mpitrace_svn20100211/Package/64
### Set the tracing configuration
export EXTRAE_CONFIG_FILE=./extrae.xml

### Dinamically load the tracing library. Don't do this if you have linked your program statically!
export LD_PRELOAD=${EXTRAE_HOME}/lib/libmpitrace.so

### Run the desired program
$*

