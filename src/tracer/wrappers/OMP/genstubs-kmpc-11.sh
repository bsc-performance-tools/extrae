#!/bin/bash

MAX_WRAPPERS=256
MAX_WRAPPERS_PER_FILE=64
INTERMEDIATE_PATH="intel-kmpc-11-intermediate"
INTERMEDIATE_WRAPPERS_BASENAME="intel-kmpc-11-intermediate-part"
INTERMEDIATE_WRAPPERS_SUFFIX=".c"
INTERMEDIATE_HEADER="${INTERMEDIATE_PATH}/intel-kmpc-11-intermediate.h"
MAX_TASKLOOP_HELPERS=1024
TASKLOOP_HELPERS_BASENAME="intel-kmpc-11-taskloop-helpers"
TASKLOOP_HELPERS_C="${INTERMEDIATE_PATH}/${TASKLOOP_HELPERS_BASENAME}.c"
TASKLOOP_HELPERS_H="${INTERMEDIATE_PATH}/${TASKLOOP_HELPERS_BASENAME}.h"

rm -f ${INTERMEDIATE_PATH}/${INTERMEDIATE_WRAPPERS_BASENAME}* ${INTERMEDIATE_HEADER} ${TASKLOOP_HELPERS_C} ${TASKLOOP_HELPERS_H}

echo "/* Automagically generated file by $0 at `date` */" > ${TASKLOOP_HELPERS_H}
echo "#ifndef __TASKLOOP_HELPERS_H__" >> ${TASKLOOP_HELPERS_H}
echo "#define __TASKLOOP_HELPERS_H__" >> ${TASKLOOP_HELPERS_H}
echo "" >> ${TASKLOOP_HELPERS_H}
echo "#include \"omp-common.h\"" >> ${TASKLOOP_HELPERS_H}
echo "" >> ${TASKLOOP_HELPERS_H}
echo "#define MAX_TASKLOOP_HELPERS ${MAX_TASKLOOP_HELPERS}" >> ${TASKLOOP_HELPERS_H}
echo "void *get_taskloop_helper_fn_ptr(int taskloop_helper_id);" >> ${TASKLOOP_HELPERS_H}
echo "" >> ${TASKLOOP_HELPERS_H}

echo "/* Automagically generated file by $0 at `date` */" > ${TASKLOOP_HELPERS_C}
echo "#include \"`basename ${TASKLOOP_HELPERS_H}`\"" >> ${TASKLOOP_HELPERS_C}
echo "#include \"intel-kmpc-11.h\"" >> ${TASKLOOP_HELPERS_C}
echo "" >> ${TASKLOOP_HELPERS_C}
echo "void *taskloop_helper_fn[${MAX_TASKLOOP_HELPERS}] = {" >> ${TASKLOOP_HELPERS_C}
for id in `seq 1 ${MAX_TASKLOOP_HELPERS}`
do
	let id=id-1
	echo -e "\ttaskloop_helper_fn_${id}," >> ${TASKLOOP_HELPERS_C}
done
echo "};" >> ${TASKLOOP_HELPERS_C}
echo "" >> ${TASKLOOP_HELPERS_C}
echo "void *get_taskloop_helper_fn_ptr(int taskloop_helper_id)" >> ${TASKLOOP_HELPERS_C}
echo "{" >> ${TASKLOOP_HELPERS_C}
echo -e "\treturn taskloop_helper_fn[taskloop_helper_id];" >> ${TASKLOOP_HELPERS_C}
echo "}" >> ${TASKLOOP_HELPERS_C}
echo "" >> ${TASKLOOP_HELPERS_C}

for id in `seq 1 ${MAX_TASKLOOP_HELPERS}`
do
	let id=id-1
	echo "void taskloop_helper_fn_${id}(int arg, void *wrap_task);" >> ${TASKLOOP_HELPERS_H}
	echo "void taskloop_helper_fn_${id}(int arg, void *wrap_task)" >> ${TASKLOOP_HELPERS_C}
	echo -e "{" >> ${TASKLOOP_HELPERS_C}
	echo -e "\thelper__kmpc_taskloop_substitute(arg, wrap_task, ${id});" >> ${TASKLOOP_HELPERS_C}
	echo -e "}" >> ${TASKLOOP_HELPERS_C}
	echo "" >> ${TASKLOOP_HELPERS_C}
done
echo "" >> ${TASKLOOP_HELPERS_H}
echo "#endif /* __TASKLOOP_HELPERS_H__ */" >> ${TASKLOOP_HELPERS_H}

echo "/* Automagically generated file by $0 at `date` */" > ${INTERMEDIATE_HEADER}
echo "#ifndef INTEL_OMP_FUNC_ENTRIES" >> ${INTERMEDIATE_HEADER}
echo "# define INTEL_OMP_FUNC_ENTRIES ${MAX_WRAPPERS}" >> ${INTERMEDIATE_HEADER}
echo "#endif" >> ${INTERMEDIATE_HEADER}

PART=0

for number in `seq 0 ${MAX_WRAPPERS}`
do
	PART=$((($number/${MAX_WRAPPERS_PER_FILE})+1))
	INTERMEDIATE_WRAPPERS="${INTERMEDIATE_PATH}/${INTERMEDIATE_WRAPPERS_BASENAME}${PART}${INTERMEDIATE_WRAPPERS_SUFFIX}"

	if [ ! -f ${INTERMEDIATE_WRAPPERS} ]; then
		echo "/* Automagically generated file by $0 at `date` */" > ${INTERMEDIATE_WRAPPERS}
		echo "#include <stdarg.h>" >> ${INTERMEDIATE_WRAPPERS}
		echo "#include <wrapper.h>" >> ${INTERMEDIATE_WRAPPERS}
		echo "#include <omp-events.h>" >> ${INTERMEDIATE_WRAPPERS}
		echo "#include \"intel-kmpc-11.h\"" >> ${INTERMEDIATE_WRAPPERS}
		echo "" >> ${INTERMEDIATE_WRAPPERS}
	fi

	# Generate the intermediate routines
	PROTOTYPE="void __kmpc_parallel_sched_${number}_args (void *p1, int p2, void *task_ptr, void *wrap_ptr, void **args)"

	echo "${PROTOTYPE};"                                                         >> ${INTERMEDIATE_HEADER}
	echo "${PROTOTYPE}"                                                          >> ${INTERMEDIATE_WRAPPERS}
	echo "{"                                                                     >> ${INTERMEDIATE_WRAPPERS}
	echo -e "\tif (wrap_ptr != NULL)"                                            >> ${INTERMEDIATE_WRAPPERS}
	echo -e -n "\t\t__kmpc_fork_call_real(p1, p2+1, wrap_ptr, task_ptr"          >> ${INTERMEDIATE_WRAPPERS}
	if [[ $number > 0 ]]; then
		for i in `seq 1 ${number}`
		do
			echo -n ", args[$(($i - 1))]"                                            >> ${INTERMEDIATE_WRAPPERS}
		done
  fi	
	echo -e ");"                                                                 >> ${INTERMEDIATE_WRAPPERS}
	echo -e "\telse"                                                             >> ${INTERMEDIATE_WRAPPERS}
	echo -e -n "\t\t__kmpc_fork_call_real(p1, p2, task_ptr"                      >> ${INTERMEDIATE_WRAPPERS}
	if [[ $number > 0 ]]; then
		for i in `seq 1 ${number}`
		do
			echo -n ", args[$(($i - 1))]"                                            >> ${INTERMEDIATE_WRAPPERS}
		done
  fi	
	echo -e ");\n}\n"                                                            >> ${INTERMEDIATE_WRAPPERS}

	PROTOTYPE="void __kmpc_parallel_wrap_${number}_args (int *p1, int *p2, void *task_ptr"
	for i in `seq 1 ${number}` 
	do
		PROTOTYPE+=", void *arg${i}"
	done
	PROTOTYPE+=")"                                                               >> ${INTERMEDIATE_WRAPPERS}

	echo "${PROTOTYPE};"                                                         >> ${INTERMEDIATE_HEADER}
	echo "${PROTOTYPE}"                                                          >> ${INTERMEDIATE_WRAPPERS}
	echo "{" >> ${INTERMEDIATE_WRAPPERS}
	echo -e -n "\tvoid (*task_real)(int*,int*"                                   >> ${INTERMEDIATE_WRAPPERS}
	for i in `seq 1 ${number}`
	do
		echo -n ",void *"                                                          >> ${INTERMEDIATE_WRAPPERS}
	done
	echo -e ") = task_ptr;\n"                                                    >> ${INTERMEDIATE_WRAPPERS}
	echo -e "\tExtrae_OpenMP_UF_Entry ((void *)task_real);"                                                            >> ${INTERMEDIATE_WRAPPERS}
	echo -e "\tBackend_Leave_Instrumentation (); /* We're entering user code */" >> ${INTERMEDIATE_WRAPPERS}
	echo -e -n "\ttask_real (p1, p2"                                             >> ${INTERMEDIATE_WRAPPERS}
	for i in `seq 1 ${number}`
	do
		echo -n ", arg${i}"                                                        >> ${INTERMEDIATE_WRAPPERS}
	done
	echo ");"                                                                    >> ${INTERMEDIATE_WRAPPERS}
	echo -e "\tExtrae_OpenMP_UF_Exit ();"                                        >> ${INTERMEDIATE_WRAPPERS}
	echo -e "}\n"                                                                >> ${INTERMEDIATE_WRAPPERS}
done

LIST_OF_PARTS=""
cp ${INTERMEDIATE_PATH}/Makefile.am.in ${INTERMEDIATE_PATH}/Makefile.am
for number in `seq 1 ${PART}`
do
	LIST_OF_PARTS+="${INTERMEDIATE_WRAPPERS_BASENAME}${number}${INTERMEDIATE_WRAPPERS_SUFFIX} "
done
sed -i "s/@LIST_OF_PARTS@/${LIST_OF_PARTS}/g" ${INTERMEDIATE_PATH}/Makefile.am

