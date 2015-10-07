# AX_SHOW_CONFIGURATION
# --------------------
# Cross compiling information:
# http://www.gnu.org/savannah-checkouts/gnu/autoconf/manual/autoconf-2.69/html_node/Hosts-and-Cross_002dCompilation.html#Hosts-and-Cross_002dCompilation 
AC_DEFUN([AX_SHOW_CONFIGURATION],
[
	if test "${host}" != "${build}" ; then
		CROSSC="${host} to ${build}"
	else
		if test "${IS_BGL_MACHINE}" = "yes" ; then
			CROSSC="${host} with BG/L system support"
		elif test "${IS_BGP_MACHINE}" = "yes" ; then
			CROSSC="${host} with BG/P system support"
		elif test "${IS_BGQ_MACHINE}" = "yes" ; then
			CROSSC="${host} with BG/Q system support"
		elif test "${IS_CRAY_XT}" = "yes" ; then
			CROSSC="${host} with Cray XT system support"
		else
			CROSSC="no"
		fi
	fi

	echo
	echo Package configuration for ${PACKAGE_NAME} ${PACKAGE_VERSION} based on ${SVN_branch} rev. ${SVN_revision}:
	echo -----------------------
	echo Installation prefix: ${prefix}
	echo Cross compilation:   ${CROSSC}
	echo CC:                  ${CC}
	echo CXX:                 ${CXX}
	echo Binary type:         ${BITS} bits
	echo 

	AX_MPI_SHOW_CONFIGURATION
	AX_OPENMP_SHOW_CONFIGURATION
	AX_OPENSHMEM_SHOW_CONFIGURATION
	AX_PTHREAD_SHOW_CONFIGURATION
	AX_CUDA_SHOW_CONFIGURATION
	AX_OPENCL_SHOW_CONFIGURATION
	AX_JAVA_SHOW_CONFIGURATION

	echo
	if test "${PMAPI_ENABLED}" = "yes" -o "${PAPI_ENABLED}" = "yes" ; then
		echo Performance counters: yes
		if test "${PMAPI_ENABLED}" = "yes" ; then
			echo -e \\\tPerformance API:  PMAPI
		else
			echo -e \\\tPerformance API:  PAPI
			echo -e \\\tPAPI home:        ${PAPI_HOME}
			echo -e \\\tSampling support: ${PAPI_SAMPLING_ENABLED}
		fi
	else
		echo Performance counters: no
	fi

	echo
	if test "${BFD_INSTALLED}" = "yes" ; then
		echo libbfd available: yes \(${BFD_LIBSDIR}\)
	else
		echo libbfd available: no
	fi

	if test "${LIBERTY_INSTALLED}" = "yes" ; then
		echo libiberty available: yes \(${LIBERTY_LIBSDIR}\)
	else
		echo libiberty available: no
	fi
	if test "${BFD_INSTALLED}" != "yes" -o "${LIBERTY_INSTALLED}" != "yes" ; then
		echo Warning! Source code addresses cannot be translated due to lack of libbfd/libiberty
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
        echo Warning! Extrae will not be able to read XML files for its configuration. Configuration will only occur through environment variables.
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
        echo On-line analysis: ${ONLINE_enabled}
        if test "${ONLINE_enabled}" = "yes" ; then
		echo -e \\\tClustering suite available: ${CLUSTERING_INSTALLED}
		echo -e \\\tSpectral analysis available: ${SPECTRAL_INSTALLED}
        fi

	if test "${USE_GETTIMEOFDAY_CLOCK}" = "yes" ; then
		echo Clock routine: gettimeofday
	else
		if test "${USE_POSIX_CLOCK}" = "yes" ; then
			if test "${NEED_POSIX_CLOCK_LIB}" = "no" ; then
				echo Clock routine: POSIX / clock_gettime, but don\'t need to link against posix clock library explicitly
			else
				echo Clock routine: POSIX / clock_gettime, library in ${POSIX_CLOCK_LIB}
			fi
		else
			echo Clock routine: low-level / architecture dependant
		fi
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
