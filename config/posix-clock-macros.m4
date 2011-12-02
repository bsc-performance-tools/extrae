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
      [enable_posix_clock="not_set"]
   )

   if test -x /sbin/lsmod ; then
     LSMOD="/sbin/lsmod"
   elif test -x /bin/lsmod ; then
     LSMOD="/bin/lsmod"
   else
     LSMOD="lsmod"
   fi
     
   if test "${enable_posix_clock}" = "not_set" ; then
      if test "${OperatingSystem}" = "linux" ; then
         acpi_cpufreq=`${LSMOD} | grep  ^acpi_cpufreq | wc -l`
         if test "${acpi_cpufreq}" -ge 1 ; then
            AC_MSG_WARN([Attention! It seems that your processor frequency changes on the fly through 'acpi_cpufreq' module. We add --enable-posix-clock to your configure line so as to use clock routines that can adapt to the processor frequency changes. However, if you know for sure that your processor speed does not change, you can proceed by adding --disable-posix-clock to use the fastest clock routines])
            enable_posix_clock="yes"
         fi
         freqtable=`${LSMOD} | grep  ^freqtable | wc -l`
         if test "${freqtable}" -ge 1 ; then
            AC_MSG_WARN([Attention! It seems that your processor frequency changes on the fly through 'freqtable' module. We add --enable-posix-clock to your configure line so as to use clock routines that can adapt to the processor frequency changes. However, if you know for sure that your processor speed does not change, you can proceed by adding --disable-posix-clock to use the fastest clock routines])
            enable_posix_clock="yes"
         fi
         upowerd=`ps -efa | grep upowerd | grep -v grep | wc -l`
         if test "${upowerd}" -ge 1 ; then
            AC_MSG_WARN([Attention! It seems that your processor frequency changes on the fly through 'upowerd'. We add --enable-posix-clock to your configure line so as to use clock routines that can adapt to the processor frequency changes. However, if you know for sure that your processor speed does not change, you can proceed by adding --disable-posix-clock to use the fastest clock routines])
            enable_posix_clock="yes"
         fi
      fi
   fi

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
      TRYING_RT_LIBS=`/sbin/ldconfig -p | grep librt | ${AWK} -F '=> ' '{ print $'2' }'`
      for ac_cv_clock_gettime_lib in "" ${TRYING_RT_LIBS} "NO" ;
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
         USE_POSIX_CLOCK="yes"
      else
         AC_MSG_RESULT([found in ${ac_cv_clock_gettime_lib}])
         USE_POSIX_CLOCK="yes"
         POSIX_CLOCK_LIB=${ac_cv_clock_gettime_lib}
         RT_LIBSDIR=`dirname ${POSIX_CLOCK_LIB}`
         RT_LDFLAGS=-L${RT_LIBSDIR}
         RT_SHAREDLIBSDIR=-L${RT_LIBSDIR}
         RT_LIBS=-lrt
      fi
   fi

  AC_SUBST(POSIX_CLOCK_LIB)
  AC_SUBST(RT_LIBSDIR)
  AC_SUBST(RT_LDFLAGS)
  AC_SUBST(RT_SHAREDLIBSDIR)
  AC_SUBST(RT_LIBS)
	AM_CONDITIONAL(USE_POSIX_CLOCK, test "${USE_POSIX_CLOCK}" = "yes")
	if test "${USE_POSIX_CLOCK}" = "yes" ; then
		AC_DEFINE([USE_POSIX_CLOCK], 1, [Defined if using posix clock routines / clock_gettime])
	fi
])

