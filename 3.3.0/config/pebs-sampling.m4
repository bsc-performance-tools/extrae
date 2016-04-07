# AX_PEBS_SAMPLING
# --------------------
AC_DEFUN([AX_PEBS_SAMPLING],
[
	AC_REQUIRE([AX_SYSTEM_TYPE])

	AC_ARG_ENABLE(pebs-sampling,
		AC_HELP_STRING(
			[--enable-pebs-sampling],
			[Allow PEBS sampling on Intel64 processors (experimental, disabled by default)]
		),
		[enable_pebs_sampling="${enableval}"],
		[enable_pebs_sampling="not_set"]
	)

	if test "${enable_pebs_sampling}" = "yes" ; then
		if test "${target_cpu}" != "amd64" -a "${target_cpu}" != "x86_64" ; then
			AC_MSG_ERROR([PEBS sampling is only available on Intel64 processors])
		fi
		if test "${OperatingSystem}" != "linux" ; then
			AC_MSG_ERROR([PEBS sampling is only available on Linux systems])
		fi
		AC_DEFINE([ENABLE_PEBS_SAMPLING], 1, [Define if PEBS sampling must be used])
	fi

	AM_CONDITIONAL(WANT_PEBS_SAMPLING, test "${enable_pebs_sampling}" = "yes")
])
