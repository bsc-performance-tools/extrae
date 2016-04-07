#!/bin/bash

rm -fr set-0 TRACE.*

EXTRAE_ON=1 ./check_Extrae_user_function
ret=$?

rm -fr set-0 TRACE.*

exit $ret
