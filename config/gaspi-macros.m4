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
		[gaspi_paths="no"]
	)

	if test "${gaspi_paths}" != "no"; then
		if test -z "${gaspi_paths}" ; then
			AC_MSG_ERROR([Attention! You have passed an invalid GASPI location.])
		fi
		if test ! -d ${gaspi_paths} ; then
			AC_MSG_ERROR([Attention! You have passed an invalid GASPI location.])
		fi
	fi

	dnl Search for GASPI installation
	AX_FIND_INSTALLATION([GASPI], [$gaspi_paths], [], [gaspi_run], [PGASPI.h], [], [GPI2], [], [], [])

	dnl Did the checks pass?
	AM_CONDITIONAL(HAVE_GASPI, test "${GASPI_INSTALLED}" = "yes")

	dnl If we have detected the GASPI launcher
	AM_CONDITIONAL(HAVE_GASPIRUN, test "${GASPI_BIN_gaspi_run}" != "")

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
		if test "${GASPI_BIN_gaspi_run}" != ""; then
		echo -e \\\tGASPI launcher:         ${GASPI_BIN_gaspi_run}
		else
		echo -e \\\tGASPI launcher was not found. It is NOT necessary to compile Extrae but it is necessary to execute the regression tests.
		fi
	fi
])
