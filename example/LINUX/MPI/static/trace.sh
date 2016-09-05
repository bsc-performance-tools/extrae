#!/bin/bash

source @sub_PREFIXDIR@/etc/extrae.sh

export EXTRAE_CONFIG_FILE=../extrae.xml

## Run the desired program
$*

