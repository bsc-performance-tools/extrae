# AX_PROG_BOOST
# --------------------
AC_DEFUN([AX_PROG_BOOST],
[
	AC_MSG_CHECKING([for boost])
	AC_ARG_WITH(boost,
		AC_HELP_STRING(
			[--with-boost@<:@=ARG@:>@],
			[Specify where boost library was installed]
		),
	[BoostDir="$withval"],
	[BoostDir="/usr"]
	)

	BOOST_enabled="no"
	if test ! -d ${BoostDir}/include/boost ; then
		AC_MSG_WARN([Could not find BOOST directory. Check for --with-boost option])
	else
		if test ! -f ${BoostDir}/include/boost/version.hpp ; then
			AC_MSG_WARN([Could not find a valid BOOST directory - missing include/boost/version.hpp. Check for --with-boost option])
		else
			BOOST_enabled="yes"
			BOOST_HOME=${BoostDir}
			AC_SUBST(BOOST_HOME)
			AC_MSG_RESULT([found in ${BoostDir}])
		fi
	fi
])
