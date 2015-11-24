#!/bin/bash

source ../../helper_functions.bash

make -f Makefile.nodebug run

exit 1

make -f Makefile.nodebug clean
