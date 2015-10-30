# AX_PROG_SIONLIB
# -----------
AC_DEFUN([AX_PROG_SIONLIB],
[
	AC_ARG_WITH(sionlib,
		AC_HELP_STRING(
			[--with-sionlib@<:@=DIR@:>@],
			[specify where to find SIONlib libraries and includes]
		),
		[sionlib_paths="$withval"],
		[sionlib_paths="not_set"] dnl List of possible default paths
	)

	if test "${sionlib_paths}" != "not_set" ; then
		dnl Search for SIONlib installation
		AX_FIND_INSTALLATION([SIONLIB], [$sionlib_paths], [sionlib])

		SIONLIB_ENABLED="yes"
		SIONLIB_HOME=${sionlib_paths}
		AC_SUBST(SIONLIB_HOME)

		if test "${SIONLIB_ENABLED}" = "yes" ; then
			AC_CHECK_HEADERS([sion.h], [], [sion_h_notfound="yes"])

			if test "${sion_h_notfound}" = "yes" ; then
				AC_MSG_ERROR([Error! Unable to find sion.h])
			fi
			SIONLIB_INSTALLED="yes"
			AC_CHECK_LIB([sionmpi_64], [sion_paropen_mpi],
				[ 
					SIONLIB_LIBS="${LIBS}"
					AC_SUBST(SIONLIB_LIBS)
				],
				[SIONLIB_ENABLED="no"]
			)
			AC_DEFINE([HAVE_SIONLIB], 1, [Define to 1 if SIONlib is installed in the system])
		fi

		SIONLIB_LIBS="-lsionmpi_64 -lsionser_64 -lsioncom_64 -lsioncom_64_lock_none"

		SIONLIB_LIBSDIR="${sionlib_paths}/libs"
		AC_SUBST(SIONLIB_LIBSDIR)
	fi
 	AM_CONDITIONAL(HAVE_SIONLIB, test "${SIONLIB_INSTALLED}" = "yes")
])

