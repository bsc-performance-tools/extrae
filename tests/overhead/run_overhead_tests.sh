#!/bin/bash

function run_test {
	echo Test `basename $1` - $2 executions
	rm -fr tmp.$$
	let total=0
	for ex in `seq $2`
	do
		# Ignore stderr!
		timing[${ex}]=`$1 2> /dev/null | grep "^RESULT :" | cut -d " " -f 4`
		echo ${timing[${ex}]} >> tmp.$$
		let total=${total}+${timing[${ex}]} 
		rm -fr set-0 TRACE.mpits TRACE.sym
	done
	min=`sort -n tmp.$$ | head -1`
	let avg=${total}/$2
	max=`sort -n tmp.$$ | tail -1`
	echo min: ${min} ns
	echo avg: ${avg} ns
	echo max: ${max} ns
	echo  # Additional line
	rm -fr tmp.$$
}

function run_test_java {
	echo Test `basename $1` - $2 executions
	rm -fr tmp.$$
	let total=0
	for ex in `seq $2`
	do
		# Ignore stderr!
		timing[${ex}]=`EXTRAE_CONFIG_FILE=extrae.xml ${EXTRAE_HOME}/bin/extraej -- $1 2> /dev/null | grep "^RESULT :" | cut -d " " -f 4`
		echo ${timing[${ex}]} >> tmp.$$
		let total=${total}+${timing[${ex}]} 
		rm -fr set-0 TRACE.mpits TRACE.sym
	done
	min=`sort -n tmp.$$ | head -1`
	let avg=${total}/$2
	max=`sort -n tmp.$$ | tail -1`
	echo min: ${min} ns
	echo avg: ${avg} ns
	echo max: ${max} ns
	echo  # Additional line
	rm -fr tmp.$$
}

function run_test_java_extraej {
	echo Test `basename $1` - $2 executions
	rm -fr tmp.$$
	let total=0
	for ex in `seq $2`
	do
		# Ignore stderr!
		timing[${ex}]=`EXTRAE_CONFIG_FILE=extraej.xml ${EXTRAE_HOME}/bin/extraej -- $1 2> /dev/null | grep "^RESULT :" | cut -d " " -f 4`
		echo ${timing[${ex}]} >> tmp.$$
		let total=${total}+${timing[${ex}]} 
		rm -fr set-0 TRACE.mpits TRACE.sym
	done
	min=`sort -n tmp.$$ | head -1`
	let avg=${total}/$2
	max=`sort -n tmp.$$ | tail -1`
	echo min: ${min} ns
	echo avg: ${avg} ns
	echo max: ${max} ns
	echo  # Additional line
	rm -fr tmp.$$
}

export EXTRAE_HOME=@sub_EXTRAE_HOME@
export EXTRAE_CONFIG_FILE=extrae.xml

EXECUTABLES="./posix_clock ./ia32_rdtsc_clock ./extrae_event ./extrae_nevent4"
EXECUTABLES+=" @sub_COUNTERS_OVERHEAD_TESTS@"
EXECUTABLES+=" @sub_CALLERS_OVERHEAD_TESTS@"
EXECUTABLES_JAVA="JavaEvent JavaNEvent4"
EXECUTABLES_JAVA_EXTRAEJ="JavaFakeRoutine"


# Compile binaries first if they do not exist

echo Checking for existing binaries, and compiling if necessary ...

for e in ${EXECUTABLES}
do
	if test ! -x ${e} ; then
		make `basename ${e}`
	fi
done
if [[ -r "${EXTRAE_HOME}/lib/javatrace.jar" ]] ; then
	for j in ${EXECUTABLES_JAVA}
	do
		make ${j}.class
	done
fi
if [[ -x "${EXTRAE_HOME}/bin/extraej" ]] ; then
	for j in ${EXECUTABLES_JAVA_EXTRAEJ}
	do
		make ${j}.class
	done
fi

echo
echo
echo ------ CUT HERE ------
echo 
echo

if [[ ! -x "${EXTRAE_HOME}/bin/extrae-header" ]] ; then
	echo Cannot locate extrae-header in ${EXTRAE_HOME}, installation corrupted!
	exit -1
else
	${EXTRAE_HOME}/bin/extrae-header
fi

echo

for e in ${EXECUTABLES}
do
	run_test ${e} 10
done

# Run Java tests?
if [[ -r "${EXTRAE_HOME}/lib/javatrace.jar" ]] ; then
	for j in ${EXECUTABLES_JAVA}
	do
	 	run_test_java ${j} 10
	done
fi
if [[ -x "${EXTRAE_HOME}/bin/extraej" ]] ; then
	for j in ${EXECUTABLES_JAVA_EXTRAEJ}
	do
	 	run_test_java_extraej ${j} 10
	done
fi
