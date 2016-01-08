# AX_PROG_BOOST
# --------------------
AC_DEFUN([AX_PROG_BOOST],
[
	BOOST_enabled="no"
	BOOST_default="no"

	AC_MSG_CHECKING([for boost])
	AC_ARG_WITH(boost,
		AC_HELP_STRING(
			[--with-boost@<:@=ARG@:>@],
			[Specify where boost library was installed]
		),
		[BoostDir="$withval"],
		[BoostDir="none"]
	)

	if test "${BoostDir}" != "none" ; then
		dnl Boost directory was given. Give it a try.
		if test ! -d ${BoostDir}/include/boost ; then
			AC_MSG_ERROR([Could not find BOOST directory. Check for --with-boost option])
		else
			if test ! -r ${BoostDir}/include/boost/version.hpp ; then
				AC_MSG_ERROR([Could not find a valid BOOST directory - missing include/boost/version.hpp. Check for --with-boost option])
			else
				BOOST_default="no"
				BOOST_enabled="yes"
				BOOST_HOME=${BoostDir}
				AC_SUBST(BOOST_HOME)
				AC_MSG_RESULT([found in ${BoostDir}])
			fi
		fi
	else
		dnl Boost directory was not given. Check if Boost are installed in the
		dnl default path
		AC_MSG_RESULT([not given --with-boost])
		AC_MSG_CHECKING([for BOOST in the default paths])
		AC_TRY_COMPILE(
			[#include <boost/version.hpp>],
			[
				unsigned version = BOOST_VERSION;
				char *version_str = BOOST_LIB_VERSION;
				return 0;
			],
			[ac_cv_boost_default="yes"],
			[ac_cv_boost_default="no"]
		)

		if test "${ac_cv_boost_default}" = "yes"; then
			BOOST_enabled="yes"
			BOOST_default="yes"
			BOOST_HOME=""
			AC_SUBST(BOOST_HOME)
			AC_MSG_RESULT([works])
		else
			AC_MSG_ERROR([Cannot find BOOST. Check for --with-boost option])
		fi
	fi

	AM_CONDITIONAL(NEED_BOOST_HOME, test "${BOOST_default}" = "no")
])
