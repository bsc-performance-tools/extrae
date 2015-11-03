#!/bin/bash

export PYTHONPATH=@sub_PREFIXDIR@/lib:$PYTHONPATH
export EXTRAE_CONFIG_FILE=./extrae.xml

python ./test.py

