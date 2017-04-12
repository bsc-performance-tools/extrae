#!/bin/bash

DOACROSS_MAX_NESTING=16
INTERMEDIATE_PATH="gnu-libgomp-intermediate"
INTERMEDIATE_DOACROSS_SWITCH="${INTERMEDIATE_PATH}/libgomp-doacross-intermediate-switch.c"

rm -f ${INTERMEDIATE_DOACROSS_SWITCH}

for number in `seq 1 ${DOACROSS_MAX_NESTING}`
do

	echo -e "\t\t\tcase ${number}:" >> ${INTERMEDIATE_DOACROSS_SWITCH}

	echo -e -n "\t\t\t\tGOMP_doacross_wait_real (first" >> ${INTERMEDIATE_DOACROSS_SWITCH}
	if [[ $number > 1 ]]; then
		for i in `seq 2 ${number}`; do
			let "tmp = i - 2"
			echo -n ", args[$tmp]" >> ${INTERMEDIATE_DOACROSS_SWITCH}
		done
	fi
	echo ");" >> ${INTERMEDIATE_DOACROSS_SWITCH}
	echo -e "\t\t\t\tbreak;" >> ${INTERMEDIATE_DOACROSS_SWITCH}
done
