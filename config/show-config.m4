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
		elif test "${IS_BGQ_MACHINE}" = "yes" ; then
			CROSSC="${host} with BG/Q system support"
		elif test "${IS_CELL_MACHINE}" = "yes" ; then
			CROSSC="${host} with Cell Broadband Engine support - SDK ${CELL_SDK}.x"
    elif test "${IS_CRAY_XT}" = "yes" ; then
      CROSSC="${host} with Cray XT system support"
		else
			CROSSC="no"
		fi
	fi

	echo
	echo Package configuration for ${PACKAGE_NAME} ${PACKAGE_VERSION}:
	echo -----------------------
	echo Installation prefix: ${prefix}
	echo Cross compilation:   ${CROSSC}
	echo CC:                  ${CC}
    if test "${GXX}" = "yes" ; then
    	echo CXX:                 ${CXX}
    else
        echo CXX:                 C++ compiler not found. MRNET, DYNINST support can not be included.
    fi
	echo Binary type:         ${BITS} bits
	echo 

	echo MPI instrumentation: ${MPI_INSTALLED}
	if test "${MPI_INSTALLED}" = "yes" ; then
		echo -e \\\tMPI home:            ${MPI_HOME}
		echo -e \\\tFortran decoration:  ${FORTRAN_DECORATION}
		echo -e \\\tperuse available?    ${PERUSE_AVAILABILITY}
		echo -e \\\tmixed C/Fortran libraries? ${mpi_lib_contains_c_and_fortran}
    echo -e \\\tshared libraries?    ${MPI_SHARED_LIB_FOUND}
		echo -e \\\t1-sided operations?  ${mpi_lib_supports_mpi_1sided}
		echo -e \\\tMPI I/O operations?  ${mpi_lib_supports_mpi_io}
	fi
	echo PACX instrumentation: ${PACX_INSTALLED}
	if test "${PACX_INSTALLED}" = "yes" ; then
		echo -e \\\tPACX home:          ${PACX_HOME}
	fi

	echo OpenMP instrumentation: ${enable_openmp}

	echo pThread instrumentation: ${enable_pthread}

	if test "${enable_cupti}" = "yes" ; then
		echo CUDA instrumentation: yes, through CUPTI
		echo -e \\\tCUDA home : ${cuda_path}
		echo -e \\\tCUPTI home: ${cupti_path}
	else
		if test "${enable_cuda}" = "yes" ; then
			echo CUDA instrumentation: yes, through LD_PRELOAD
			echo -e \\\tCUDA home : ${cuda_path}
		else
			echo CUDA instrumentation: no
		fi
	fi

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
	if test "${BFD_INSTALLED}" = "yes" ; then
		echo libbfd available: yes \(${BFD_HOME}\)
	fi

	if test "${LIBERTY_INSTALLED}" = "yes" ; then
		echo libiberty available: yes \(${LIBERTY_HOME}\)
	else
		echo libiberty available: no
	fi

	if test "${zhome_dir}" != "not found" ; then
		echo zlib available: yes \(${LIBZ_HOME}\)
	else
		echo zlib available: no
	fi

	if test "${XML_enabled}" = "yes" ; then
		echo libxml2 available: yes \(${XML2_HOME}\)
	else
		echo libxml2 available: no
	fi

	if test "${BOOST_enabled}" = "yes" ; then
		if test "${BOOST_default}" = "yes" ; then
			echo BOOST available: yes \(in default compiler path\)
		else
			echo BOOST available: yes \(${BOOST_HOME}\)
		fi
	else
		echo BOOST available: no
	fi

	if test "${libunwind_works}" = "yes" ; then
		echo -e callstack access: through libunwind \(${UNWIND_HOME}\)
	else
		if test "${OperatingSystem}" = "linux" -a "${Architecture}" = "ia64" ; then
			echo callstack access: libunwind required for Linux/ia64 !
		elif test "${OperatingSystem}" = "linux" -a "${Architecture}" != "ia64" ; then
			echo "callstack access: through backtrace (from linux)"
		else
			echo callstack access: none
		fi
	fi

	echo
	if test "${DYNINST_HOME}" != "" ; then
		echo Dynamic instrumentation: yes \(${DYNINST_HOME}\)
	else
		echo Dynamic instrumentation: no
	fi

	echo
 	echo Optional features:
	echo ------------------
  if test "${USE_POSIX_CLOCK}" = "yes" ; then
    echo Clock routine: POSIX / clock_gettime
  else
    echo Clock routine: low-level / architecture dependant
  fi
 	echo Heterogeneous support: ${enable_hetero}
	if test "${MPI_INSTALLED}" = "yes" -a "${enable_parallel_merge}" = "yes" ; then
		echo Parallel merge: yes
	else
		if test "${MPI_INSTALLED}" != "yes" ; then
			echo Parallel merge: not available as MPI is not given
		else
			echo Parallel merge: disabled
		fi
	fi
])
