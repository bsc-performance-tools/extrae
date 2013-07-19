#!/bin/bash

function run_test {
	echo Test `basename $1` - $2 executions taking the $3 best runs
	rm -fr tmp.$$
	for ex in `seq $2`
	do
		# Ignore stderr!
		$1 2>/dev/null | grep "^RESULT :" | cut -d " " -f 4- >> tmp.$$
		rm -fr set-0 TRACE.mpits TRACE.sym
	done
	sort -n tmp.$$ | head -$3
	echo  # Additional line
	rm -fr tmp.$$
}


export EXTRAE_CONFIG_FILE=extrae.xml
export LD_LIBRARY_PATH+=:@sub_EXTRAE_HOME@/lib:@sub_PAPI_SHAREDLIBSDIR@:@sub_UNWIND_SHAREDLIBSDIR@

EXECUTABLES="./posix_clock ./ia32_rdtsc_clock ./extrae_event ./extrae_nevent4"
EXECUTABLES+=" @sub_COUNTERS_OVERHEAD_TESTS@"
EXECUTABLES+=" @sub_CALLERS_OVERHEAD_TESTS@"

# Compile binaries first if they do not exist

echo Checking for existing binaries, and compiling if necessary ...

for e in ${EXECUTABLES}
do
	if test ! -x ${e} ; then
		make `basename ${e}`
	fi
done

echo
echo
echo ------ CUT HERE ------
echo 
echo

if test ! -x @sub_EXTRAE_HOME@/bin/extrae-header ; then
	echo Cannot locate extrae-header in @sub_EXTRAE_HOME@, installation corrupted!
	exit -1
else
	@sub_EXTRAE_HOME@/bin/extrae-header
fi

echo

for e in ${EXECUTABLES}
do
	run_test $e 10 3
done
