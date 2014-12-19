# AX_JAVA
# --------------------
AC_DEFUN([AX_JAVA],
[
	AC_ARG_WITH(java,
		AC_HELP_STRING(
			[--with-java@<:@=DIR@:>@],
			[Enable support for tracing Java (experimental)]
		),
		[java_path="${withval}"],
		[java_path="none"]
	)

	if test -z "${java_path}" ; then
		AC_MSG_ERROR([Cannot find JAVA])
	fi

	if test "${java_path}" != "none" ; then
		AC_MSG_CHECKING([for Java directory])
		if test -d "${java_path}" ; then
			AC_MSG_RESULT([found])
	
			AC_MSG_CHECKING([for Java compiler])
			if test -x "${java_path}/bin/javac" ; then
				AC_MSG_RESULT(${java_path}/bin/javac)
				JAVAC="${java_path}/bin/javac"
				JAVAC_found="yes"
			else
				AC_MSG_ERROR([Java compiler was not found])
			fi
	
			AC_MSG_CHECKING([for Java header and stub file generator])
			if test -x "${java_path}/bin/javah" ; then
				AC_MSG_RESULT(${java_path}/bin/javah)
				JAVAH="${java_path}/bin/javah"
				JAVAH_found="yes"
			else
				AC_MSG_ERROR([Java header and stub file generator was not found])
			fi

			AC_MSG_CHECKING([for Java archive tool])
			if test -x "${java_path}/bin/jar" ; then
				AC_MSG_RESULT(${java_path}/bin/jar)
				JAR="${java_path}/bin/jar"
				JAR_found="yes"
			else
				AC_MSG_RESULT([Java archive tool was not found])
			fi

			AC_MSG_CHECKING([for Java include directories])
			if test ! -d "${java_path}/include" ; then
				AC_MSG_ERROR([Cannot find include directory])
			fi
			if test "${OperatingSystem}" = "linux" ; then
				if test ! -d "${java_path}/include/linux" ; then
					AC_MSG_ERROR([Cannot find linux/linux directory])
				fi
			fi
			AC_MSG_RESULT(found)
		fi

		JAVA_found="yes"
		JAVA_path=${java_path}
		JAVA_INCLUDES="-I${JAVA_path}/include"

		if test "${OperatingSystem}" = "linux" ; then
			JAVA_INCLUDES="${JAVA_INCLUDES} -I${JAVA_path}/include/linux"
		elif test "${OperatingSystem}" = "darwin" ; then
			JAVA_INCLUDES="${JAVA_INCLUDES} -I/System/Library/Frameworks/JavaVM.framework/Versions/Current/Headers"
		fi
	fi

	AM_CONDITIONAL(WANT_JAVA, test "${JAVA_found}" = "yes")
	AC_SUBST(JAVAC)
	AC_SUBST(JAVAH)
	AC_SUBST(JAR)
	AC_SUBST(JAVA_INCLUDES)
])

# AX_JAVA_SHOW
# --------------------
AC_DEFUN([AX_JAVA_SHOW_CONFIGURATION],
[
	if test "${JAVA_found}" = "yes" ; then
		echo Java instrumentation: supported \(${JAVA_path}\)
	else
		echo Java instrumentation: unsupported
	fi
])
