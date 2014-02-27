# AX_OPENMP
# --------------------
AC_DEFUN([AX_CHECK_OPENMP],
[
	AC_ARG_ENABLE(openmp,
	   AC_HELP_STRING(
	      [--enable-openmp],
	      [Enable support for tracing OpenMP -Intel, IBM and GNU runtimes- (enabled by default, IBM only on PPC)]
	   ),
	   [enable_openmp="${enableval}"],
	   [enable_openmp="yes"]
	)

	if test "${enable_openmp}" = "yes" ; then
		AX_OPENMP()

		AC_ARG_ENABLE(openmp-intel,
		   AC_HELP_STRING(
		      [--enable-openmp-intel],
		      [Enable support for tracing OpenMP Intel]
		   ),
		   [enable_openmp_intel="${enableval}"],
		   [enable_openmp_intel="yes"]
		)
		AC_ARG_ENABLE(openmp-gnu,
		   AC_HELP_STRING(
		      [--enable-openmp-gnu],
		      [Enable support for tracing OpenMP GNU]
		   ),
		   [enable_openmp_gnu="${enableval}"],
		   [enable_openmp_gnu="yes"]
		)
		AC_ARG_ENABLE(openmp-ibm,
		   AC_HELP_STRING(
		      [--enable-openmp-ibm],
		      [Enable support for tracing OpenMP IBM (only on PPC)]
		   ),
		   [enable_openmp_ibm="${enableval}"],
		   [enable_openmp_ibm="yes"]
		)
	fi

	if test "${enable_openmp_intel}" = "yes" -o \
	        "${enable_openmp_gnu}" = "yes" -o \
	        "${enable_openmp_gnu}" = "yes" ; then
		enable_openmp="yes"
	else
		enable_openmp="no"
	fi

	AM_CONDITIONAL(WANT_OPENMP, test "${enable_openmp}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_INTEL, test "${enable_openmp_intel}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_GNU, test "${enable_openmp_gnu}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_IBM, test "${enable_openmp_ibm}" = "yes" )
])

# AX_OPENMP_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_OPENMP_SHOW_CONFIGURATION],
[
	if test "${enable_openmp}" = "yes" ; then
		echo OpenMP instrumentation: yes, through LD_PRELOAD
		echo -e \\\tGNU OpenMP: ${enable_openmp_gnu}
		echo -e \\\tIBM OpenMP: ${enable_openmp_ibm}
		echo -e \\\tIntel OpenMP: ${enable_openmp_intel}
	else
		echo OpenMP instrumentation: no
  fi
])
