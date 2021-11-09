# AX_PROG_OPENACC
# -----------
AC_DEFUN([AX_PROG_OPENACC],
[
	AX_FLAGS_SAVE()

	AC_ARG_WITH(openacc,
		AC_HELP_STRING(
			[--with-openacc@<:@=DIR@:>@],
			[specify where to find OPENACC libraries and includes]
		),
		[openacc_paths=${withval}],
		[openacc_paths="no"]
	)

	if test "${openacc_paths}" = "not_set" ; then
		AC_MSG_ERROR([Attention! You have not specified the location of the OPENACC library through --with-openacc option. Please, use --with-openacc to specify the location of the OPENACC installation on your system, or if you don't want Extrae to support OPENACC instrumentation use --without-openacc instead.])
	fi

	if test "${openacc_paths}" != "no"; then
		if test -z "${openacc_paths}" ; then
			AC_MSG_ERROR([Attention! You have passed an invalid OPENACC location.])
		fi
		if test ! -d ${openacc_paths} ; then
			AC_MSG_ERROR([Attention! You have passed an invalid OPENACC location.])
		fi
	fi

	dnl Search for OPENACC installation
	AX_FIND_INSTALLATION([OPENACC], [$openacc_paths], [], [], [], [], [], [], [], [])

	if test "${OPENACC_INSTALLED}" = "yes" ; then
		if test -d "${OPENACC_INCLUDES}/openacc" ; then
			OPENACC_INCLUDES="${OPENACC_INCLUDES}/openacc"
			OPENACC_CFLAGS="-I${OPENACC_INCLUDES}"
			CPPFLAGS="${OPENACC_CFLAGS} ${CPPFLAGS}"
		fi

		AC_CHECK_HEADERS([acc_prof.h], [], [OPENACC_INSTALLED="no"])

		if test ${OPENACC_INSTALLED} = "no" ; then
			AC_MSG_ERROR([Couldn't find acc_prof.h file in the OPENACC specified path.])
		fi

		AC_MSG_CHECKING([for OPENACC library])

		if test -f "${OPENACC_LIBSDIR_MULTIARCH}/libacchost.so"; then
			OPENACC_LIBSDIR=${OPENACC_LIBSDIR_MULTIARCH}
		fi

		if test -f "${OPENACC_LIBSDIR}/libacchost.so"; then
			OPENACC_LIBS="-lacchost"
		fi

		AC_MSG_RESULT([${OPENACC_LIBSDIR}, ${OPENACC_LIBS}])

		if test "${OPENACC_LIBS}" = "not found" ; then
			AC_MSG_ERROR([Couldn't find OPENACC libraries file in the OPENACC specified path.])
		fi
	fi

	AC_SUBST(OPENACC_LIBSDIR)
	AC_SUBST(OPENACC_LIBS)
	AC_SUBST(OPENACC_CFLAGS)

	dnl Did the checks pass?
	AM_CONDITIONAL(WANT_OPENACC, test "${OPENACC_INSTALLED}" = "yes")

	if test "${OPENACC_INSTALLED}" = "yes" ; then
		AC_DEFINE([WANT_OPENACC], 1, [Determine if OPENACC in installed])
	fi

	AX_FLAGS_RESTORE()
])

# AX_OPENACC_SHOW_CONFIGURATION
# ----------
AC_DEFUN([AX_OPENACC_SHOW_CONFIGURATION],
[
	echo OPENACC instrumentation: ${OPENACC_INSTALLED}
	if test "${OPENACC_INSTALLED}" = "yes" ; then
		echo -e \\\tOPENACC home:             ${OPENACC_HOME}
	fi
])
