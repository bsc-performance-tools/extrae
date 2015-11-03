# AX_CHECK_GETTIMEOFDAY_CLOCK
# -------------------
AC_DEFUN([AX_CHECK_GETTIMEOFDAY_CLOCK],
[
   USE_GETTIMEOFDAY_CLOCK="no"
   AC_ARG_ENABLE(gettimeofday-clock,
      AC_HELP_STRING(
         [--enable-gettimeofday-clock],
         [Use gettimeofday clock instead of low level timing routines (disabled by default)]
      ),
      [enable_gettimeofday_clock="${enableval}"],
      [enable_gettimeofday_clock="not_set"]
   )
   USE_GETTIMEOFDAY_CLOCK=${enable_gettimeofday_clock}

   AM_CONDITIONAL(USE_GETTIMEOFDAY_CLOCK, test "${USE_GETTIMEOFDAY_CLOCK}" = "yes")
   if test "${USE_GETTIMEOFDAY_CLOCK}" = "yes" ; then
      AC_DEFINE([USE_GETTIMEOFDAY_CLOCK], 1, [Defined if using gettimeofday clock routine])
   fi
])

