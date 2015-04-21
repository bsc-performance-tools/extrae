# AX_OPENMP
# --------------------
AC_DEFUN([AX_CHECK_OPENMP],
[
	AC_REQUIRE([AX_HAVE_SYNC_FETCH_AND_ADD])

	AC_ARG_ENABLE(openmp,
	   AC_HELP_STRING(
	      [--enable-openmp],
	      [Enable support for tracing OpenMP -Intel, IBM and GNU runtimes- (enabled by default)]
	   ),
	   [enable_openmp="${enableval}"],
	   [enable_openmp="yes"]
	)

	if test "${enable_openmp}" = "yes" ; then
		AX_OPENMP()

		AC_ARG_ENABLE(openmp-ompt,
		   AC_HELP_STRING(
		      [--enable-openmp-ompt],
		      [Enable support for tracing OpenMP through OMPT interface]
		   ),
		   [enable_openmp_ompt="${enableval}"],
		   [enable_openmp_ompt="no"]
		)

		# OMP & other OpenMP instrumentations are not compatible
		if test "${enable_openmp_ompt}" = "no" ; then
			if test "${Architecture}" = "ia32"; then
				enable_openmp_intel_default="yes"
			else
				enable_openmp_intel_default="no"
			fi
			AC_ARG_ENABLE(openmp-intel,
			   AC_HELP_STRING(
			      [--enable-openmp-intel],
			      [Enable support for tracing OpenMP Intel (enabled by default on Intel systems)]
			   ),
			   [enable_openmp_intel="${enableval}"],
			   [enable_openmp_intel="${enable_openmp_intel_default}"]
			)
			AC_ARG_ENABLE(openmp-gnu,
			   AC_HELP_STRING(
			      [--enable-openmp-gnu],
			      [Enable support for tracing OpenMP GNU]
			   ),
			   [enable_openmp_gnu="${enableval}"],
			   [enable_openmp_gnu="yes"]
			)
			if test "${Architecture}" = "powerpc"; then
				enable_openmp_ibm_default="yes"
			else
				enable_openmp_ibm_default="no"
			fi
			AC_ARG_ENABLE(openmp-ibm,
			   AC_HELP_STRING(
			      [--enable-openmp-ibm],
			      [Enable support for tracing OpenMP IBM (enabled by default on PowerPC systems)]
			   ),
			   [enable_openmp_ibm="${enableval}"],
			   [enable_openmp_ibm="${enable_openmp_ibm_default}"]
			)

		else
			enable_openmp_ibm="no"
			enable_openmp_gnu="no"
			enable_openmp_intel="no"
			enable_openmp_ompt="yes"

			AC_DEFINE([OMPT_INSTRUMENTATION], [1], [Define if OpenMP is instrumented through OMPT])
		fi

	fi

	if test "${enable_openmp_intel}" = "yes" -o \
	        "${enable_openmp_gnu}" = "yes" -o \
	        "${enable_openmp_ibm}" = "yes" -o \
	        "${enable_openmp_ompt}" = "yes" ; then
		enable_openmp="yes"
	else
		enable_openmp="no"
	fi

	if test "${enable_openmp}" = "yes" -a "${enable_openmp_gnu}" = "yes"; then
		AC_ARG_WITH([libgomp-version],
			AC_HELP_STRING(
				[--with-libgomp-version@<:@=ARG@:>@],
				[Specify version compatibility with libgomp. Valid values are: 4.2, 4.9, auto (default)]
			),
			[libgomp_version="$withval"],
			[libgomp_version="auto"]
		)
		if test "${libgomp_version}" = "auto"; then
			AC_MSG_CHECKING([for libgomp version based on the compiler version])
			if test "${ax_cv_c_compiler_vendor}" != "gnu" ; then
				AC_MSG_ERROR([Cannot detect libgomp version from C compiler ($CC)])
			else
				if test ${_ax_c_compiler_version_major} -ge 5; then
				  libgomp_version="4.9"
				elif test ${_ax_c_compiler_version_major} -eq 4 -a ${_ax_c_compiler_version_minor} -ge 9; then
				  libgomp_version="4.9"
				elif test ${_ax_c_compiler_version_major} -eq 4 -a ${_ax_c_compiler_version_minor} -ge 2; then
				  libgomp_version="4.2"
				else
				  AC_MSG_ERROR([C compiler does not seem to include libgomp, version too old?])
				fi
				AC_MSG_RESULT([${libgomp_version}])
			fi
		elif test "${libgomp_version}" != "4.2" -a \
		          "${libgomp_version}" != "4.9"; then
			AC_MSG_ERROR([Invalid libgomp version '$libgomp_version'. Valid values for --with-libgomp_version are: 4.2, 4.9, auto (default)])
		fi
	fi

	AM_CONDITIONAL(WANT_OPENMP, test "${enable_openmp}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_INTEL, test "${enable_openmp_intel}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_GNU, test "${enable_openmp_gnu}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_IBM, test "${enable_openmp_ibm}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_OMPT, test "${enable_openmp_ompt}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_GNU_4_2, test "${libgomp_version}" = "4.2" )
	AM_CONDITIONAL(WANT_OPENMP_GNU_4_9, test "${libgomp_version}" = "4.9" )
])

AC_DEFUN([AX_HAVE_SYNC_FETCH_AND_ADD],
[
	AC_MSG_CHECKING([for __sync_fetch_and_add availability])
	AC_TRY_LINK(
		[ ], 
		[ volatile int i; __sync_fetch_and_add(&i,1); ],
		[ have_sync_fetch_and_add="yes" ]
	)

	if test "${have_sync_fetch_and_add}" = "yes" ; then
		AC_DEFINE([HAVE__SYNC_FETCH_AND_ADD], 1, [Define if __sync_fetch_and_add is available])
		AC_MSG_RESULT([yes])
	else
		AC_MSG_RESULT([no])
	fi
])

# AX_OPENMP_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_OPENMP_SHOW_CONFIGURATION],
[
	if test "${enable_openmp}" = "yes" ; then
		echo OpenMP instrumentation: yes, through LD_PRELOAD
		if test "${enable_openmp_gnu}" = "yes"; then
			echo -e \\\tGNU OpenMP: yes, libgomp ${libgomp_version}
		else
			echo -e \\\tGNU OpenMP: no
		fi
		echo -e \\\tIBM OpenMP: ${enable_openmp_ibm}
		echo -e \\\tIntel OpenMP: ${enable_openmp_intel}
		echo -e \\\tOMPT: ${enable_openmp_ompt}
	else
		echo OpenMP instrumentation: no
  fi
])

