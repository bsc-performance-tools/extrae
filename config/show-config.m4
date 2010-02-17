# AX_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_SHOW_CONFIGURATION],
[
	if test "${host}" != "${target}" ; then
		CROSSC="${host} to ${target}"
	else
		if test "${IS_BGL_MACHINE}" = "yes" ; then
			CROSSC="${host} with BG/L system support"
		elif test "${IS_BGP_MACHINE}" = "yes" ; then
			CROSSC="${host} with BG/P system support"
		elif test "${IS_CELL_MACHINE}" = "yes" ; then
			CROSSC="${host} with Cell Broadband Engine support - SDK ${CELL_SDK}.x"
		else
			CROSSC="no"
		fi
	fi

	echo
	echo Package configuration :
	echo -----------------------
	echo Installation prefix: ${prefix}
	echo Cross compilation:   ${CROSSC}
	echo CC:                  ${CC}
	echo CXX:                 ${CXX}
	echo Binary type:         ${BITS} bits
	echo 

	echo MPI instrumentation: ${MPI_INSTALLED}
	if test "${MPI_INSTALLED}" = "yes" ; then
		echo -e \\\tFortran decoration:  ${FORTRAN_DECORATION}
		echo -e \\\tMPI home:            ${MPI_HOME}
		echo -e \\\tperuse available?    ${PERUSE_AVAILABILITY}
		echo -e \\\tmixed C/Fortran libraries? ${mpi_lib_contains_c_and_fortran}
    echo -e \\\tshared libraries?    ${MPI_SHARED_LIB_FOUND}
	fi
	echo
	echo PACX instrumentation: ${PACX_INSTALLED}
	if test "${PACX_INSTALLED}" = "yes" ; then
		echo -e \\\tPACX home:          ${PACX_HOME}
	fi

	echo
	echo OpenMP instrumentation: ${enable_openmp}

	echo
	echo pThread instrumentation: ${enable_pthread}

	echo
	if test "${PMAPI_ENABLED}" = "yes" -o "${PAPI_ENABLED}" = "yes" ; then
		echo Performance counters at instrumentation: yes
		if test "${PMAPI_ENABLED}" = "yes" ; then
			echo -e \\\tPerformance API:  PMAPI
		else
			echo -e \\\tPerformance API:  PAPI
			echo -e \\\tPAPI home:        ${PAPI_HOME}
			echo -e \\\tSampling support: ${PAPI_SAMPLING_ENABLED}
		fi
	else
		echo Performance counters at instrumentation: no
	fi

	echo
	echo BFD availability: ${BFD_INSTALLED}
	if test "${BFD_INSTALLED}" = "yes" ; then
		echo -e \\\tBFD home:  ${BFD_HOME}
	fi

	echo
	echo liberty availability: ${LIBERTY_INSTALLED}
	if test "${LIBERTY_INSTALLED}" = "yes" ; then
		echo -e \\\tliberty home:  ${LIBERTY_HOME}
	fi

	echo
	if test "${DYNINST_HOME}" != "" ; then
		echo Dynamic instrumentation: yes
		echo -e \\\tDynInst home: ${DYNINST_HOME}
	else
		echo Dynamic instrumentation: no
	fi

	echo
 	echo Optional features:
	echo ------------------
 	echo Heterogeneous support: ${enable_hetero}
	echo Parallel merge: ${enable_parallel_merge}
])
