#!/bin/bash

source ../helper_functions.bash
source ../../../etc/extrae.sh 

FILES=`echo ../../../example/LINUX/MPI/*xml`

if [[ "${FILES}" = "" ]]; then
	die "Cannot find XML files in ../../../example/LINUX/MPI"
fi
if [[ "${FILES}" = "../../../example/LINUX/MPI/*xml" ]]; then
	die "Cannot find XML files in ../../../example/LINUX/MPI"
fi

for f in ${FILES}
do
	echo Testing ${f}
	EXTRAE_CONFIG_FILE=${f} ./check_Extrae_xml
	if [[ "${?}" -ne 0 ]]; then
		die "Error parsing ${f}"
	fi
done

rm -fr TRACE.* set-0

exit 0
