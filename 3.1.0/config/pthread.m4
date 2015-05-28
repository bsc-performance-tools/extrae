# AX_PTHREAD
# --------------------
AC_DEFUN([AX_PTHREAD],
[
	AC_ARG_ENABLE(pthread,
		AC_HELP_STRING(
			[--enable-pthread],
			[Enable support for tracing pthread library (enabled by default)]
		),
		[enable_pthread="${enableval}"],
		[enable_pthread="yes"]
	)
	if test "${enable_pthread}" = "yes" ; then
		ACX_PTHREAD([],[AC_MSG_ERROR([Unable to determine pthread library support])])
	fi

	if test "${enable_pthread}" = "yes" ; then

		AX_FLAGS_SAVE()

		AC_MSG_CHECKING([whether pthread_create exists])
		CFLAGS="${CFLAGS} ${PTHREAD_CFLAGS}"
		LIBS="${LIBS} ${PTHREAD_LIBS}"
		AC_TRY_LINK(
			[ #include <pthread.h> ], 
			[ int r = pthread_create ((pthread_t*)0, (pthread_attr_t*)0, (void*)main, (void*)0); ],
			[ pthread_create_exists="yes" ]
		)
		AC_MSG_RESULT([${pthread_create_exists}])

		if test "${pthread_create_exists}" != "yes" ; then
			AC_MSG_ERROR([Cannot create pthread-based applications! See config.log for further details.])
		fi

		AC_MSG_CHECKING([whether pthread_barrier_wait exists])
		AC_TRY_LINK(
			[ #include <pthread.h> ], 
			[ int r = pthread_barrier_wait ((pthread_barrier_t*)0); ],
			[ pthread_barrier_wait_exists="yes" ]
		)
		AC_MSG_RESULT([${pthread_barrier_wait_exists}])

		AX_FLAGS_RESTORE()

		if test "${pthread_barrier_wait_exists}" = "yes" ; then
			AC_DEFINE([HAVE_PTHREAD_BARRIER_WAIT], [1], [Determine if pthread_barrier_wait exists and can be instrumented])
		else
			pthread_barrier_wait_exists="no"
		fi
	fi

	AM_CONDITIONAL(WANT_PTHREAD, test "${enable_pthread}" = "yes" )

	AC_ARG_ENABLE(pthread-support-in-all-libs,
		AC_HELP_STRING(
			[--enable-pthread-support-in-all-libs],
			[Allows all the instrumentation libraries to work with pthreads. Caution! May add dependencies with pthread library (disabled by default)]
		),
		[enable_pthread_in_all_libs="${enableval}"],
		[enable_pthread_in_all_libs="no"]
	)
	AM_CONDITIONAL(PTHREAD_SUPPORT_IN_ALL_LIBS, test "${enable_pthread_in_all_libs}" = "yes" -a "${enable_pthread}" = "yes" )

])


# AX_PTHREAD_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_PTHREAD_SHOW_CONFIGURATION],
[
	if test "${enable_pthread}" = "yes" -a "${pthread_create_exists}" = "yes" ; then
		echo pThread instrumentation: yes
		echo -e \\\tSupport for pthread_barrier_wait: ${pthread_barrier_wait_exists}
	else
		echo pThread instrumentation: no
	fi
])
