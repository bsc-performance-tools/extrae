#!/bin/bash

rm -fr *.sym *.mpits set-0

EXTRAE_ON=1 ./ompss-codelocation

# Actual comparison
diff ompss-codelocation.reference set-0/*.sym
