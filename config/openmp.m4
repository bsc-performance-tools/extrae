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

		AC_ARG_WITH(openmp-ompt,
		   AC_HELP_STRING(
		      [--with-openmp-ompt=<header>],
		      [Enable support for tracing OpenMP through OMPT interface - giving OMPT header file]
		   ),
		   [with_openmp_ompt="${withval}"],
		   [with_openmp_ompt="no"]
		)

		# OMP & other OpenMP instrumentations are not compatible

		if test "${with_openmp_ompt}" = "no" ; then

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

		else
			enable_openmp_ibm="no"
			enable_openmp_gnu="no"
			enable_openmp_intel="no"

			if test ! -r ${with_openmp_ompt} ; then
				AC_MSG_ERROR([Cannot find OMPT header given by --with-openmp-ompt=])
			fi

			enable_openmp_ompt="yes"

			AC_DEFINE([OMPT_INSTRUMENTATION], [1], [Define if OpenMP is instrumented through OMPT])
			AC_DEFINE_UNQUOTED([OMPT_HEADER_LOCATION], "${with_openmp_ompt}", [Location of the OMPT header file])
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

	AM_CONDITIONAL(WANT_OPENMP, test "${enable_openmp}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_INTEL, test "${enable_openmp_intel}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_GNU, test "${enable_openmp_gnu}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_IBM, test "${enable_openmp_ibm}" = "yes" )
	AM_CONDITIONAL(WANT_OPENMP_OMPT, test "${enable_openmp_ompt}" = "yes" )
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
		echo -e \\\tOMPT: ${enable_openmp_ompt}
		if test "${enable_openmp_ompt}" ; then
			echo -e \\\t\\\tHeader file: ${with_openmp_ompt}
		fi
	else
		echo OpenMP instrumentation: no
  fi
])
