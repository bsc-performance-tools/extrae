# AX_PROG_GPI
# -----------
AC_DEFUN([AX_PROG_GPI],
[
	AX_FLAGS_SAVE()

	AC_ARG_WITH(gpi,
		AC_HELP_STRING(
			[--with-gpi@<:@=DIR@:>@],
			[specify where to find GPI libraries and includes]
		),
		[gpi_paths=${withval}],
		[gpi_paths="not_set"]
	)

	if test "${mpi_paths}" = "not_set" ; then
		AC_MSG_ERROR([Attention! You have not specified the location of the GPI library through --with-mpi option. Please, use --with-gpi to specify the location of the GPI installation on your system, or if you don't want Extrae to support GPI instrumentation use --without-gpi instead.])
	fi

	if test "${gpi_paths}" != "no"; then
		if test -z "${gpi_paths}" ; then
			AC_MSG_ERROR([Attention! You have passed an invalid GPI location.])
		fi
		if test ! -d ${gpi_paths} ; then
			AC_MSG_ERROR([Attention! You have passed an invalid GPI location.])
		fi
	fi

	dnl Search for GPI installation
	AX_FIND_INSTALLATION([GPI], [$gpi_paths], [gpi])

	if test "${GPI_INSTALLED}" = "yes" ; then
		if test -d "${GPI_INCLUDES}/gpi" ; then
			GPI_INCLUDES="${GPI_INCLUDES}/gpi"
			GPI_CFLAGS="-I${GPI_INCLUDES}"
			CPPFLAGS="${GPI_CFLAGS} ${CPPFLAGS}"
		fi

		AC_CHECK_HEADERS([PGASPI.h], [], [GPI_INSTALLED="no"])

		if test ${GPI_INSTALLED} = "no" ; then
			AC_MSG_ERROR([Couldn't find PGASPI.h file in the GPI specified path.])
		fi

		AC_MSG_CHECKING([for GPI library])

		if test -f "${MPI_LIBSDIR_MULTIARCH}/libGPI2.so"; then 
			GPI_LIBSDIR="${GPI_LIBSDIR_MULTIARCH}"
			GPI_LIBS="-lGPI2"
		fi

		AC_MSG_RESULT([${GPI_LIBSDIR}, ${MPI_LIBS}])

		if test "${GPI_LIBS}" = "not found" ; then
			AC_MSG_ERROR([Couldn't find GPI libraries file in the GPI specified path.])
		fi
	fi

	AC_MSG_CHECKING([for MPI launcher])
	GPIRUN=""
	for gpix in [ "gaspi_run" ]; do
		if test -x "${GPI_HOME}/bin${BITS}/${gpix}" ; then
			MPIRUN="${GPI_HOME}/bin${BITS}/${gpix}"
			break
		elif test -x "${GPI_HOME}/bin/${gpix}" ; then
			GPIRUN="${GPI_HOME}/bin/${gpix}"
			break
		fi
	done
	if test "${GPIRUN}" != "" ; then
		AC_MSG_RESULT([${GPIRUN}])
	else
		AC_MSG_RESULT([not found! -- It is not needed to compile Extrae but it is necessary to execute regression tests])
	fi

	AC_SUBST(GPIRUN)

	dnl Did the checks pass?
	AM_CONDITIONAL(HAVE_GPI, test "${GPI_INSTALLED}" = "yes")

	dnl If we have detected the GPI launcher
	AM_CONDITIONAL(HAVE_GPIRUN, test "${GPIRUN}" != "")

	if test "${GPI_INSTALLED}" = "yes" ; then
		AC_DEFINE([HAVE_GPI], 1, [Determine if GPI in installed])
	fi

	AX_FLAGS_RESTORE()
])

# AX_GPI_SHOW_CONFIGURATION
# ----------
AC_DEFUN([AX_GPI_SHOW_CONFIGURATION],
[
	echo GPI instrumentation: ${GPI_INSTALLED}
	if test "${GPI_INSTALLED}" = "yes" ; then
		echo -e \\\tGPI home:             ${GPI_HOME}
		if test "${GPIRUN}" != ""; then
		echo -e \\\tGPI launcher:         ${GPIRUN}
		else
		echo -e \\\tGPI launcher was not found. It is NOT necessary to compile Extrae but it is necessary to execute the regression tests.
		fi
	fi
])
