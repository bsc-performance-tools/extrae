# AX_PROG_GASPI
# -----------
AC_DEFUN([AX_PROG_GASPI],
[
	AX_FLAGS_SAVE()

	AC_ARG_WITH(gaspi,
		AC_HELP_STRING(
			[--with-gaspi@<:@=DIR@:>@],
			[specify where to find GASPI libraries and includes]
		),
		[gaspi_paths=${withval}],
		[gaspi_paths="not_set"]
	)

	if test "${mpi_paths}" = "not_set" ; then
		AC_MSG_ERROR([Attention! You have not specified the location of the GASPI library through --with-mpi option. Please, use --with-gaspi to specify the location of the GASPI installation on your system, or if you don't want Extrae to support GASPI instrumentation use --without-gaspi instead.])
	fi

	if test "${gaspi_paths}" != "no"; then
		if test -z "${gaspi_paths}" ; then
			AC_MSG_ERROR([Attention! You have passed an invalid GASPI location.])
		fi
		if test ! -d ${gaspi_paths} ; then
			AC_MSG_ERROR([Attention! You have passed an invalid GASPI location.])
		fi
	fi

	dnl Search for GASPI installation
	AX_FIND_INSTALLATION([GASPI], [$gaspi_paths], [gaspi])

	if test "${GASPI_INSTALLED}" = "yes" ; then
		if test -d "${GASPI_INCLUDES}/gaspi" ; then
			GASPI_INCLUDES="${GASPI_INCLUDES}/gaspi"
			GASPI_CFLAGS="-I${GASPI_INCLUDES}"
			CPPFLAGS="${GASPI_CFLAGS} ${CPPFLAGS}"
		fi

		AC_CHECK_HEADERS([PGASPI.h], [], [GASPI_INSTALLED="no"])

		if test ${GASPI_INSTALLED} = "no" ; then
			AC_MSG_ERROR([Couldn't find PGASPI.h file in the GASPI specified path.])
		fi

		AC_MSG_CHECKING([for GASPI library])

		if test -f "${GASPI_LIBSDIR_MULTIARCH}/libGPI2.so"; then 
			GASPI_LIBSDIR="${GASPI_LIBSDIR_MULTIARCH}"
			GASPI_LIBS="-lGPI2"
		fi

		AC_MSG_RESULT([${GASPI_LIBSDIR}, ${GASPI_LIBS}])

		if test "${GASPI_LIBS}" = "not found" ; then
			AC_MSG_ERROR([Couldn't find GASPI libraries file in the GASPI specified path.])
		fi
	fi

	AC_MSG_CHECKING([for GASPI launcher])
	GASPIRUN=""
	for gaspix in [ "gaspi_run" ]; do
		if test -x "${GASPI_HOME}/bin${BITS}/${gaspix}" ; then
			GASPIRUN="${GASPI_HOME}/bin${BITS}/${gaspix}"
			break
		elif test -x "${GASPI_HOME}/bin/${gaspix}" ; then
			GASPIRUN="${GASPI_HOME}/bin/${gaspix}"
			break
		fi
	done
	if test "${GASPIRUN}" != "" ; then
		AC_MSG_RESULT([${GASPIRUN}])
	else
		AC_MSG_RESULT([not found! -- It is not needed to compile Extrae but it is necessary to execute regression tests])
	fi

	AC_SUBST(GASPIRUN)

	dnl Did the checks pass?
	AM_CONDITIONAL(HAVE_GASPI, test "${GASPI_INSTALLED}" = "yes")

	dnl If we have detected the GASPI launcher
	AM_CONDITIONAL(HAVE_GASPIRUN, test "${GASPIRUN}" != "")

	if test "${GASPI_INSTALLED}" = "yes" ; then
		AC_DEFINE([HAVE_GASPI], 1, [Determine if GASPI in installed])
	fi

	AX_FLAGS_RESTORE()
])

# AX_GASPI_SHOW_CONFIGURATION
# ----------
AC_DEFUN([AX_GASPI_SHOW_CONFIGURATION],
[
	echo GASPI instrumentation: ${GASPI_INSTALLED}
	if test "${GASPI_INSTALLED}" = "yes" ; then
		echo -e \\\tGASPI home:             ${GASPI_HOME}
		if test "${GASPIRUN}" != ""; then
		echo -e \\\tGASPI launcher:         ${GASPIRUN}
		else
		echo -e \\\tGASPI launcher was not found. It is NOT necessary to compile Extrae but it is necessary to execute the regression tests.
		fi
	fi
])
