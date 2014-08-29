#!/bin/bash

rm -fr *.sym *.mpits set-0

EXTRAE_ON=1 ./ompss-codelocation

grep -v ^B set-0/*.sym > OUTPUT

# Actual comparison
diff ompss-codelocation.reference OUTPUT
