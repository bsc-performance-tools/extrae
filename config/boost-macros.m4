# AX_PROG_BOOST
# --------------------
AC_DEFUN([AX_PROG_BOOST],
[
	BOOST_enabled="no"
	BOOST_default="no"

  dnl Check for boost in default paths
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
	AC_MSG_RESULT([${ac_cv_boost_default}])

	if test "${ac_cv_boost_default}" == "yes" ; then

		BOOST_enabled="yes"
		BOOST_default="yes"
		BOOST_HOME=""
		AC_SUBST(BOOST_HOME)

	else

		AC_MSG_CHECKING([for boost])
		AC_ARG_WITH(boost,
			AC_HELP_STRING(
				[--with-boost@<:@=ARG@:>@],
				[Specify where boost library was installed]
			),
		[BoostDir="$withval"],
		[BoostDir="/usr"]
		)

		if test ! -d ${BoostDir}/include/boost ; then
			AC_MSG_WARN([Could not find BOOST directory. Check for --with-boost option])
		else
			if test ! -f ${BoostDir}/include/boost/version.hpp ; then
				AC_MSG_WARN([Could not find a valid BOOST directory - missing include/boost/version.hpp. Check for --with-boost option])
			else
				BOOST_default="no"
				BOOST_enabled="yes"
				BOOST_HOME=${BoostDir}
				AC_SUBST(BOOST_HOME)
				AC_MSG_RESULT([found in ${BoostDir}])
			fi
		fi

	fi

	AM_CONDITIONAL(NEED_BOOST_HOME, test "${BOOST_default}" = "no")
])
