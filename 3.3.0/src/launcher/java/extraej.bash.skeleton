#!/bin/bash

die() { echo "$@" 1>&2 ; exit 1; }

#
# Helper functions to parse XML file
#

read_dom () {
    local IFS=\>
    read -d \< ENTITY CONTENT
    local RET=$?
    TAG_NAME=${ENTITY%% *}
    IFS=' ' read -a ATTRIBUTES <<< "${ENTITY#* }"
    return $RET
}

parse_dom () {
    if [[ "${TAG_NAME}" = "user-functions" ]]; then
		for attribute in "${ATTRIBUTES[@]}"
		do
			local IFS='='
			local pair
			read -a pair <<< "${attribute}"
			if [[ "${pair[0]}" = "enabled" ]]; then
				if [[ "${pair[1]//\"/}" = "yes" ]]; then  # remove " (all) from attribute
					extrae_function_list_enabled="yes"
				fi
			fi
			if [[ "${pair[0]}" = "list" ]]; then
				local file=${pair[1]//\"/} # remove " (all) from the attribute
			fi
		done
		if [[ "${extrae_function_list_enabled}" == "yes" ]]; then
			if [[ "${file}" != "" ]] ; then
				if [[ -r "${file}" ]]; then
					extrae_function_list_file=${file}
				else
					die "Cannot access file ${file}"
				fi
			fi
		fi
    fi
}

parse_xml () {
	while read_dom; do
	    parse_dom
		if [[ "${extrae_function_list_enabled}" = "yes" ]]; then
			if [[ "${extrae_function_list_file}" != "" ]]; then
				parse_function_list_file ${extrae_function_list_file}
				break
			fi
		fi
	done < ${1}
}

#
# Helper function to parse function list. Check pairs <class, function>
#

parse_function_list_file () {
	while read line; do
		local IFS=' '
		local entry
		read -a entry <<< "${line}"
		if [[ "${#entry[@]}" -gt 1 ]] ; then
			echo "Warning! Line is badly formatted: ${line}. Will only instrument ${entry[0]}"
		fi
		if [[ "${#entry[@]}" -ge 1 ]] ; then
			if [[ "${entry[0]}" =~ .*\*.* ]] ; then
				echo "Warning! Ignoring entry ${entry[0]} because it contains invalid characters"
			elif [[ "${entry[0]}" = "main" ]] ; then
				echo "Warning! Ignoring entry main. It is automatically instrumented"
			else
				echo "Instrumenting '${entry[0]}'"
				MemberArray+=(${entry[0]})
			fi
		fi
	done < ${1}
}

#
# Helper function to generate aspects file
#

generate_aspects () {
	mkdir ${tmpdir}/aspects
	local pa=${tmpdir}/aspects/ExtraeAspects.java
	echo "/* File automatically generated */" > ${pa}
	echo "package extrae.aspects;" >> ${pa}
	echo >> ${pa}
	echo "import org.aspectj.lang.reflect.*;" >> ${pa}
	echo >> ${pa}
	echo "public aspect ExtraeAspects" >> ${pa}
	echo "{" >> ${pa}
	echo >> ${pa}
	echo "  SourceLocation[] m_SourceLocations;" >> ${pa}
	for i in ${!MemberArray[@]}
	do 
		local index=$((i+16)) # addresses 0 & 1 are already reserved
		local member=${MemberArray[$i]}
		echo >> ${pa}
		echo "  pointcut   ${member//./_}_execution() : !within(ExtraeAspects) && execution (* ${member} (..)); " >> ${pa}
		echo "  before() : ${member//./_}_execution()" >> ${pa}
		echo "  { " >> ${pa}
		echo "    es.bsc.cepbatools.extrae.Wrapper.functionEventFromAddress (${index});" >> ${pa}
		echo "    if (m_SourceLocations[${index}] == null)" >> ${pa}
		echo "      m_SourceLocations[${index}] = thisJoinPoint.getSourceLocation();" >> ${pa}
		echo "  }" >> ${pa}
		echo "  after()  : ${member//./_}_execution()" >> ${pa}
		echo "  { es.bsc.cepbatools.extrae.Wrapper.functionEventFromAddress (0); }" >> ${pa}
	done

	echo >> ${pa}
	echo >> ${pa}
	echo "  /* main symbol instrumentation and adds symbolic information to the instrumented routines */" >> ${pa}
	echo "  pointcut main_execution() : execution ( public static void main (..));" >> ${pa}
	echo "  before () : main_execution()" >> ${pa}
	echo "  {" >> ${pa}
	echo "    /* es.bsc.cepbatools.extrae.Wrapper.Init(); not needed, auto instrumentation of main? */ " >> ${pa}
	echo "    m_SourceLocations = new SourceLocation[16+${#MemberArray[@]}];" >> ${pa}
	echo "  }" >> ${pa}
	echo "  after  () : main_execution()" >> ${pa}
	echo "  {" >> ${pa}
	echo "    /* es.bsc.cepbatools.extrae.Wrapper.Fini(); not needed, auto instrumentation of main? */" >> ${pa}
	for i in ${!MemberArray[@]}
	do 
		local index=$((i+16)) # addresses 0 & 1 are already reserved
		local member=${MemberArray[$i]}
		echo "    if (m_SourceLocations[${index}] != null)" >> ${pa}
		echo "      es.bsc.cepbatools.extrae.Wrapper.registerFunctionAddres (" >> ${pa}
		echo "        ${index}," >> ${pa}
		echo "        \"${member}\", " >> ${pa}
		echo "        m_SourceLocations[${index}].getFileName()," >> ${pa}
		echo "        m_SourceLocations[${index}].getLine());" >> ${pa}
	done
	echo "  }" >> ${pa}
	echo "}" >> ${pa} 
}

#
# Parse parameters
#

parse_parameters () {
	while [ "${1+defined}" ]; do
		if [[ "${1}" = "-keep" ]]; then
			keep_instrumented_files="yes"
		elif [[ "${1}" = "-v" ]]; then
			verbose="yes"
		elif [[ "${1}" = "-reuse" ]] ; then
			shift
			reuse_dir=${1}
		elif [[ "${1}" = "--" ]]; then
			application_separator_found="yes"
			break
		fi
	  shift
	done
}

#
# Execute Java
#

execute_java () {
	local cp=${EXTRAE_HOME}/lib/javatrace.jar:$1
	shift
	local preload=$1
	shift

	# Check whether Extrae supported JVMTI
	if [[ ! -r ${EXTRAE_HOME}/lib/libextrae-jvmti-agent.so ]]; then
		if [[ "${verbose}" = "yes" ]]; then
			echo "Executing: LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:\${LD_LIBRARY_PATH} LD_PRELOAD=${preload} CLASSPATH=${cp} ${JAVA} ${@}"
		fi

		LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:${LD_LIBRARY_PATH} \
		LD_PRELOAD=${preload} \
		CLASSPATH=${cp} \
		  ${JAVA} ${@}
	else
		local agent=${EXTRAE_HOME}/lib/libextrae-jvmti-agent.so

		if [[ "${verbose}" = "yes" ]]; then
			echo "Executing: LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:\${LD_LIBRARY_PATH} LD_PRELOAD=${preload} CLASSPATH=${cp} ${JAVA} -agentpath:${EXTRAE_HOME}/lib/libjavajvmtitrace.so ${@}"
		fi

		LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:${LD_LIBRARY_PATH} \
		LD_PRELOAD=${preload} \
		CLASSPATH=${cp} \
		  ${JAVA} -agentpath:${EXTRAE_HOME}/lib/libextrae-jvmti-agent.so ${@}
	fi
}


#
# Main entry point
#

# global variables 
extrae_function_list_file=""
extrae_function_list_enabled=""
MemberArray=()
keep_instrumented_files="no"
verbose="no"
reuse_dir=""
application_separator_found="no"

# Check for configuration

JAVA=@sub_JAVA@
AJC=@sub_AJC@
ASPECTWEAVER_JAR=@sub_ASPECTWEAVER_JAR@
EXTRAE_HOME=@sub_PREFIXDIR@

# Check existance for config file pointed by EXTRAE_CONFIG_FILE

if [[ "${#}" -eq 0 ]]; then
	echo "Usage for Extrae/Java-based instrumenter:"
	echo ""
	echo "  "`basename ${0}`" {options} -- <JavaMethod>"
	echo ""
	echo "  where options:"
	echo "    -v           Makes execution verbose"
    echo "    -keep        Runs and saves the instrumented package for a future use"
    echo "    -reuse <dir> Tells "`basename ${0}`" to reuse a previously instrumented package saved through -keep"
	echo
	echo " Example:"
	echo "  \$\{EXTRAE_HOME\}/bin/"`basename ${0}`" -v -keep -- PiExample"
	echo
	echo " Note: Remember to indicate the configuration file through the EXTRAE_CONFIG_FILE environment variable"
	die ""
fi

if [[ "${EXTRAE_CONFIG_FILE}" != "" ]]; then
	if [[ ! -r "${EXTRAE_CONFIG_FILE}" ]]; then
		die "Error! Cannot access file pointed by EXTRAE_CONFIG_FILE (${EXTRAE_CONFIG_FILE})"
	fi
else
	die "Error! EXTRAE_CONFIG_FILE was not set"
fi

parse_parameters "${@}"

if [[ "${application_separator_found}" = "no" ]]; then
	die "You need to specify the Java execution class after --"
fi

#
# Skip parameters until -- is found
#
for p in ${@}
do
	if [[ "${p}" = "--" ]]; then
		shift
		break
	fi
	shift
done

# Do we support AspectJ?
if [[ -x "${AJC}" ]]; then

	# Do we have to reuse a previous existing instrumented package?
	if [[ "${reuse_dir}" = "" ]]; then

		parse_xml ${EXTRAE_CONFIG_FILE}

		if [[ ${#MemberArray[@]} -gt 0 ]]; then

			tmpdir=`mktemp -d extraej.XXXXXX`

			generate_aspects
		
			if [[ "${verbose}" = "yes" ]]; then
				echo "Executing: CLASSPATH=${ASPECTWEAVER_JAR}:${EXTRAE_HOME}/lib/javatrace.jar:${CLASPATH} ${AJC} -inpath . -sourceroots ${tmpdir}/aspects -d ${tmpdir}/instrumented"
			fi
		
			CLASSPATH=${ASPECTWEAVER_JAR}:${EXTRAE_HOME}/lib/javatrace.jar:${CLASPATH} \
			${AJC} \
			 -inpath . \
			 -sourceroots ${tmpdir}/aspects \
			 -d ${tmpdir}/instrumented
			if [[ "${?}" -ne 0 ]]; then
				die "Error! ${AJC} failed"
			fi

			execute_java ${tmpdir}/instrumented:${ASPECTWEAVER_JAR}:${CLASSPATH} \
				${EXTRAE_HOME}/lib/libpttrace.so \
				"$@"
		else

			execute_java "${CLASSPATH}" \
				${EXTRAE_HOME}/lib/libpttrace.so \
				"$@"
		fi
	else
		# We want to reuse a previousl instrumented app
		if [[ -d "${reuse_dir}" ]]; then
			if [[ -d "${reuse_dir}/instrumented" ]]; then
				echo "Reusing instrumented application from ${reuse_dir}"
			else
				die "Cannot previously instrumented files within ${reuse_dir}/instrumented"
			fi
		else
			die "Cannot find previously instrumented application in ${reuse_dir}"
		fi

		# Let's execute the code with Extrae support
		execute_java ${reuse_dir}/instrumented:${ASPECTWEAVER_JAR}:${CLASSPATH} \
			${EXTRAE_HOME}/lib/libpttrace.so \
			"$@"
	fi
else

	# We don't support AJC. Let's execute the code with Extrae support
	execute_java "${CLASSPATH}" \
		${EXTRAE_HOME}/lib/libpttrace.so \
		"$@"
fi

if [[ "${?}" -eq 0 ]]; then
	if [[ "${keep_instrumented_files}" != "yes" ]]; then
		rm -fr ${tmpdir}
	fi
fi

