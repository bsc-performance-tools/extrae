#!/bin/bash
export EXTRAE_CONFIG_FILE=extrae.xml
export NX_ARGS="${NX_ARGS} --instrumentation=extrae "

$*
