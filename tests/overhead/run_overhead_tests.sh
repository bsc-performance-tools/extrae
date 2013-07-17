#!/bin/bash

function run_test {
	echo Running test $1 - $2 executions taking the $3 best runs
	rm -fr tmp.$$
	for ex in `seq $2`
	do
		# Ignore stderr!
		$1 2>/dev/null | grep "^RESULT :" | cut -d " " -f 4- >> tmp.$$
		rm -fr set-0 TRACE.mpits TRACE.sym
	done
	sort -n tmp.$$ | head -$3
	rm -fr tmp.$$
}


export EXTRAE_CONFIG_FILE=extrae.xml
export LD_LIBRARY_PATH+=:@sub_EXTRAE_HOME@/lib:@sub_PAPI_SHAREDLIBSDIR@

EXECUTABLES="./extrae_event ./extrae_eventandcounters ./extrae_user_function ./extrae_get_caller1 ./extrae_get_caller4 ./extrae_trace_callers ./papi_read1 ./papi_read4"

for e in ${EXECUTABLES}
do
	run_test $e 10 3
done
