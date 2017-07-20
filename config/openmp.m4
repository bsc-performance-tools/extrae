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

		# Enable GNU support
		AC_ARG_ENABLE(openmp-gnu,
		   AC_HELP_STRING(
		      [--enable-openmp-gnu],
		      [Enable support for tracing OpenMP GNU (enabled by default)]
		   ),
		   [enable_openmp_gnu="${enableval}"],
		   [enable_openmp_gnu="yes"]
		)
	
		# Enable Intel support
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

		# Enable IBM support
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

		# Enable OMPT support
		AC_ARG_ENABLE(openmp-ompt,
		   AC_HELP_STRING(
		      [--enable-openmp-ompt],
		      [Enable support for tracing OpenMP through OMPT interface (enabled by default)]
		   ),
		   [enable_openmp_ompt="${enableval}"],
		   [enable_openmp_ompt="yes"]
		)

		if test "$enable_openmp_ompt" = "yes"; then
			AC_DEFINE([OMPT_SUPPORT], [1], [Define if OMPT support is enabled])
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
			echo -e \\\tGNU OpenMP: yes
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

