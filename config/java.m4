# AX_JAVA
# --------------------
AC_DEFUN([AX_JAVA],
[
	AC_REQUIRE([AX_PTHREAD])

	AC_ARG_WITH(java-jdk,
		AC_HELP_STRING(
			[--with-java-jdk@<:@=DIR@:>@],
			[Enable support for tracing Java throug the given Java JDK (experimental)]
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

			AC_MSG_CHECKING([for Java])
			if test -x "${java_path}/bin/java" ; then
				AC_MSG_RESULT(${java_path}/bin/java)
				JAVA="${java_path}/bin/java"
				JAVA_found="yes"
			else
				AC_MSG_ERROR([Java was not found])
			fi
	
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

			AC_MSG_CHECKING([for Java JVMTI header files])
			if test -r "${java_path}/include/jvmti.h"; then
				AC_MSG_RESULT([found])
				JVMTI_found="yes"
			else
				AC_MSG_RESULT([not found])
				JVMTI_found="no"
			fi
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

	if test "${JAVA_found}" = "yes" ; then
		if test "${enable_pthread}" = "no" ; then
			AC_MSG_ERROR([Java support requires pthread support. Add --enable-pthread into the configure line.])
		fi
	fi

	AM_CONDITIONAL(WANT_JAVA, test "${JAVA_found}" = "yes")
	AM_CONDITIONAL(WANT_JAVA_JVMTI, test "${JAVA_found}" = "yes" -a "${JVMTI_found}" = "yes")
	AC_SUBST(JAVA)
	AC_SUBST(JAVAC)
	AC_SUBST(JAVAH)
	AC_SUBST(JAR)
	AC_SUBST(JAVA_INCLUDES)
])

# AX_JAVA_ASPECTJ
# --------------------
AC_DEFUN([AX_JAVA_ASPECTJ],
[
	AC_REQUIRE([AX_JAVA])
	
	AC_ARG_WITH(java-aspectj,
		AC_HELP_STRING(
			[--with-java-aspectj@<:@=DIR@:>@],
			[Enable support for tracing Java through AspectJ (experimental) -- also need --with-java-jdk and --with-java-aspectj-weaver]
		),
		[aspectj_path="${withval}"],
		[aspectj_path="none"]
	)

	aspectj_ajc_found="no"
	if test "${aspectj_path}" != "none" ; then
		AC_MSG_CHECKING([for AspectJ directory])
		if test -d "${aspectj_path}" ; then
			AC_MSG_RESULT([found])
			AC_MSG_CHECKING([for AspectJ compiler (ajc)])
			if test -x "${aspectj_path}/bin/ajc" ; then
				AJC="${aspectj_path}/bin/ajc"
				AC_MSG_RESULT([found])
				aspectj_ajc_found="yes"
			else
				AC_MSG_ERROR([Cannot find ajc within AspectJ given path!])
			fi
		else
			AC_MSG_ERROR([Given path for AspectJ does not exist!])
		fi
	fi

	AC_ARG_WITH(java-aspectj-weaver,
		AC_HELP_STRING(
			[--with-java-aspectj-weaver@<:@=DIR@:>@],
			[Enable support for tracing Java through AspectJ (experimental) -- also need --with-java-jdk and --with-java-aspectj. Indicates where to find aspectjweaver.jar.]
		),
		[aspectj_weaver_path="${withval}"],
		[aspectj_weaver_path="none"]
	)

	if test "${aspectj_ajc_found}" = "yes"; then
		aspect_weaver_found="no"

		if test "${aspectj_weaver_path}" = "none" ; then
			# Try to automatically locate aspectweaver.jar based from ASPECTJ directory
			AC_MSG_NOTICE([--with-aspect-weaver was not given. Trying to automatically locate aspectweaver.jar])
			if test -r "${aspectj_path}/share/java/aspectjweaver.jar" ; then
				aspectj_weaver_path=${aspectj_path}/share/java
			fi
		fi

		if test "${aspectj_weaver_path}" != "none" ; then
			AC_MSG_CHECKING([for AspectJ weaver directory])
			if test -d "${aspectj_weaver_path}" ; then
				AC_MSG_RESULT([found])
				AC_MSG_CHECKING([for AspectJ aspectjweaver.jar])
				if test -r "${aspectj_weaver_path}/aspectjweaver.jar"; then
					AC_MSG_RESULT([found])
					AC_MSG_CHECKING([for AspectJ aspectjweaver.jar contents])
					${JAR} tf ${aspectj_weaver_path}/aspectjweaver.jar > /dev/null
					if test "${?}" -eq 0 ; then
						AC_MSG_RESULT([seems correct])
						ASPECT_WEAVER_JAR="${aspectj_weaver_path}/aspectjweaver.jar"
						aspectj_weaver_found="yes"
					else
						AC_MSG_ERROR([the aspectjweaver.jar file does not seem valid])
					fi
				else
					AC_MSG_ERROR([Cannot find aspectjweaver.jar in the given AspectJ weaver directory])
				fi
			else
				AC_MSG_ERROR([Given path for AspectJ weaver file does not exist!])
			fi
		fi

		if test "${aspectj_weaver_found}" = "no"; then
			AC_MSG_ERROR([AspectJ requires AspectJ weaver .jar file but it was not given or located. Did you provide a valid --with-aspectj-weaver?])
		fi
	fi

	if test "${aspectj_ajc_found}" = "yes" -a "${aspectj_weaver_found}" = "yes" ; then
		ASPECTJ_found="yes"
	fi

	AM_CONDITIONAL(WANT_JAVA_WITH_ASPECTJ, test "${ASPECTJ_found}" = "yes")
	AC_SUBST(AJC)
	AC_SUBST(ASPECT_WEAVER_JAR)
])


# AX_JAVA_SHOW
# --------------------
AC_DEFUN([AX_JAVA_SHOW_CONFIGURATION],
[
	if test "${JAVA_found}" = "yes" ; then
		echo Java instrumentation: supported \(${JAVA_path}\)
		echo -e \\\tJVMTI support: ${JVMTI_found}
		if test "${ASPECTJ_found}" = "yes" ; then
			echo -e \\\tAspectJ support: yes
			echo -e \\\tAspectJ compiler: ${AJC}
			echo -e \\\tAspectJ weaver: ${ASPECT_WEAVER_JAR}
		else
			echo -e \\\tAspectJ support: no
		fi
	else
		echo Java instrumentation: unsupported
	fi
])
