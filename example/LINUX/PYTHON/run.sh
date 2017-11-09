#!/bin/bash

export PYTHONPATH=@sub_PREFIXDIR@/libexec:$PYTHONPATH
export EXTRAE_CONFIG_FILE=./extrae.xml

python ./test.py

