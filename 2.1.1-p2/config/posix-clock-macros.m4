# AX_CHECK_POSIX_CLOCK
# -------------------
AC_DEFUN([AX_CHECK_POSIX_CLOCK],
[
   USE_POSIX_CLOCK="no"
   AC_ARG_ENABLE(posix-clock,
      AC_HELP_STRING(
         [--enable-posix-clock],
         [Use POSIX clock (clock_gettime call) instead of low level timing routines (disabled by default)]
      ),
      [enable_posix_clock="${enableval}"],
      [enable_posix_clock="no"]
   )

   if test "${enable_posix_clock}" = "yes"; then

      dnl Check for POSIX C source level first
      AC_MSG_CHECKING([for _POSIX_C_SOURCE to be at least 199309L])
      AC_TRY_COMPILE(
         [#include <time.h>],
         [
            #if _POSIX_C_SOURCE < 199309L
            # error "POSIX_C_SOURCE is lower than 199309L!"
            #endif
            return 0;
         ],
         [ac_cv_posix_csource_199309L="yes"],
         [ac_cv_posix_csource_199309L="no"]
      )
      AC_MSG_RESULT([${ac_cv_posix_csource_199309L}])
      if test "${ac_cv_posix_csource_199309L}" = "no" ; then
         AC_MSG_ERROR([Cannot proceed with --enable-posix-clock and _POSIX_C_SOURCE < 199309L])
      fi

      dnl Check for _POSIX_MONOTONIC_CLOCK in unistd.h
      AC_MSG_CHECKING([for _POSIX_MONOTONIC_CLOCK to be defined])
      AC_TRY_COMPILE(
         [#include <unistd.h>],
         [
            #ifndef _POSIX_MONOTONIC_CLOCK
            # error "_POSIX_MONOTONIC_CLOCK is undefined!"
            #endif
            return 0;
         ],
         [ac_cv_posix_motonic_clock_defined="yes"],
         [ac_cv_posix_motonic_clock_defined="no"]
      )
      AC_MSG_RESULT([${ac_cv_posix_motonic_clock_defined}])
      if test "${ac_cv_posix_motonic_clock_defined}" = "no" ; then
         AC_MSG_ERROR([Cannot proceed because posix monotonic clock seems to be not available])
      fi

      dnl Check for existance of clock_gettime / CLOCK_MONOTONIC
      AC_MSG_CHECKING([for clock_gettime and CLOCK_MONOTONIC in libraries])
      LIBS_old=${LIBS}
      for ac_cv_clock_gettime_lib in "" "-lrt" "NO" ;
      do
         LIBS="${ac_cv_clock_gettime_lib}"
         AC_TRY_LINK(
            [#include <time.h>], 
            [
               struct timespec t;
               clock_gettime (CLOCK_MONOTONIC, &t);
            ],
            [break],
            []
         )
      done
      LIBS=${LIBS_old}
      if test "${ac_cv_clock_gettime_lib}" = "NO" ; then
         AC_MSG_ERROR([Unable to find a library that contains clock_gettime (typically found in -lc or -lrt)])
      fi
      if test "${ac_cv_clock_gettime_lib}" = "" ; then
         AC_MSG_RESULT([found])
      else
         AC_MSG_RESULT([found in ${ac_cv_clock_gettime_lib}])
      fi
      USE_POSIX_CLOCK="yes"
			POSIX_CLOCK_LIB=${ac_cv_clock_gettime_lib}
   fi

  AC_SUBST(POSIX_CLOCK_LIB)
	AM_CONDITIONAL(USE_POSIX_CLOCK, test "${USE_POSIX_CLOCK}" = "yes")
	if test "${USE_POSIX_CLOCK}" = "yes" ; then
		AC_DEFINE([USE_POSIX_CLOCK], 1, [Defined if using posix clock routines / clock_gettime])
	fi
])

