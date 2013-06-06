# AX_CHECK_PACX_STATUS_SIZE
# ---------------------
AC_DEFUN([AX_CHECK_PACX_STATUS_SIZE],
[
   AC_MSG_CHECKING([for size of the PACX_Status struct])
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${PACX_INCLUDES}"
   AC_TRY_RUN(
      [
         #include <pacx.h>
         int main()
         {
            return sizeof(PACX_Status)/sizeof(int);
         }
      ],
      [ SIZEOF_PACX_STATUS="0" ],
      [ SIZEOF_PACX_STATUS="$?"]
   )
   AC_MSG_RESULT([${SIZEOF_PACX_STATUS}])
   AC_DEFINE_UNQUOTED([SIZEOF_PACX_STATUS], ${SIZEOF_PACX_STATUS}, [Size of the PACX_Status structure in "sizeof-int" terms])
   AX_FLAGS_RESTORE()
])

# AX_CHECK_PACX_SOURCE_OFFSET
#------------------------
AC_DEFUN([AX_CHECK_PACX_SOURCE_OFFSET],
[
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${PACX_INCLUDES}"

   AC_CHECK_MEMBER(PACX_Status.MPI_SOURCE,,
                [AC_MSG_ERROR([We need PACX_Status.MPI_SOURCE!])],
                [#include <pacx.h>])

   AC_MSG_CHECKING([for offset of MPI_SOURCE field in PACX_Status])
   AC_TRY_RUN(
      [
         #include <pacx.h>
         int main()
         {
            PACX_Status s;
            long addr1 = (long) &s;
            long addr2 = (long) &(s.MPI_SOURCE);

            return (addr2 - addr1)/sizeof(int);
         }
      ],
      [ PACX_SOURCE_OFFSET="0" ],
      [ PACX_SOURCE_OFFSET="$?"]
   )
   AC_MSG_RESULT([${PACX_SOURCE_OFFSET}])
   AC_DEFINE_UNQUOTED([PACX_SOURCE_OFFSET], ${PACX_SOURCE_OFFSET}, [Offset of the SOURCE field in PACX_Status in sizeof-int terms])
   AX_FLAGS_RESTORE()
])

# AX_CHECK_PACX_TAG_OFFSET
#------------------------
AC_DEFUN([AX_CHECK_PACX_TAG_OFFSET],
[
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${PACX_INCLUDES}"

   AC_CHECK_MEMBER(PACX_Status.MPI_TAG,,
                [AC_MSG_ERROR([We need PACX_Status.MPI_TAG!])],
                [#include <pacx.h>])

   AC_MSG_CHECKING([for offset of TAG field in PACX_Status])
   AC_TRY_RUN(
      [
         #include <pacx.h>
         int main()
         {
            PACX_Status s;
            long addr1 = (long) &s;
            long addr2 = (long) &(s.MPI_TAG);

            return (addr2 - addr1)/sizeof(int);
         }
      ],
      [ PACX_TAG_OFFSET="0" ],
      [ PACX_TAG_OFFSET="$?"]
   )
   AC_MSG_RESULT([${PACX_TAG_OFFSET}])
   AC_DEFINE_UNQUOTED([PACX_TAG_OFFSET], ${PACX_TAG_OFFSET}, [Offset of the TAG field in PACX_Status in sizeof-int terms])
   AX_FLAGS_RESTORE()
])


# AX_PROG_PACX
# -----------
AC_DEFUN([AX_PROG_PACX],
[
   AC_REQUIRE([AX_PROG_MPI])

   AX_FLAGS_SAVE()

   AC_ARG_WITH(pacx,
      AC_HELP_STRING(
         [--with-pacx@<:@=DIR@:>@],
         [specify where to find PACX libraries and includes]
      ),
      [pacx_paths=${withval}],
      [pacx_paths="none"]
   )

   dnl Search for PACX installation
   AX_FIND_INSTALLATION([PACX], [${pacx_paths}], [pacx])

   if test "${PACX_INSTALLED}" = "yes" ; then

      dnl Check whether MPI is given at configure step
	    if test ${MPI_INSTALLED} != "yes" ; then
         AC_MSG_ERROR([PACX instrumentation requires MPI instrumentation. Add MPI instrumentation by using --with-mpi= parameter])
      fi

      dnl Check for the PACX header files.
      AC_CHECK_HEADERS([pacx.h], [], [PACX_INSTALLED="no"])

      dnl Check for the PACX library.
      dnl We won't use neither AC_CHECK_LIB nor AC_TRY_LINK because this library may have unresolved references to other libs (i.e: libgm).
      AC_MSG_CHECKING([for PACX library])
      if test -f "${PACX_LIBSDIR}/libpacx.a" -o -f "${PACX_LIBSDIR}/libpacx.so" ; then
         PACX_LIBS="-lpacx"
      else
         PACX_LIBS="not found"
      fi
      AC_MSG_RESULT([${PACX_LIBSDIR} , ${PACX_LIBS}])

			AC_MSG_CHECKING([for fortran PACX library])
      if test -f "${PACX_LIBSDIR}/libpacxf.a" -o -f "${PACX_LIBSDIR}/libpacxf.so" ; then
         PACX_F_LIB_FOUND="yes"
         PACX_F_LIB="-lpacxf"
      else
         PACX_F_LIB_FOUND="not found"
         PACX_F_LIB="not found"
      fi
			AC_MSG_RESULT([${PACX_F_LIB_FOUND}])

      if test "${PACX_LIBSDIR}" = "not found" ; then
         PACX_INSTALLED="no"
      else
         PACX_LDFLAGS="${PACX_LDFLAGS}"
         AC_SUBST(PACX_LDFLAGS)
         AC_SUBST(PACX_LIBS)
      fi
   fi

   dnl Did the checks pass?
   AM_CONDITIONAL(HAVE_PACX, test "${PACX_INSTALLED}" = "yes")

   if test "${PACX_INSTALLED}" = "yes" ; then
      AC_DEFINE([HAVE_PACX], 1, [Determine if PACX in installed])
   fi

   AX_FLAGS_RESTORE()
])

# AX_CHECK_PPACX_NAME_MANGLING
# ---------------------------
AC_DEFUN([AX_CHECK_PPACX_NAME_MANGLING],
[
   AC_REQUIRE([AX_PROG_PACX])

   AC_ARG_WITH(pacx-name-mangling,
      AC_HELP_STRING(
         [--with-pacx-name-mangling@<:@=ARG@:>@], 
         [choose the name decoration scheme for external Fortran symbols in PACX library from: 0u, 1u, 2u, upcase, auto @<:@default=auto@:>@]
      ),
      [name_mangling="$withval"],
      [name_mangling="auto"]
   )

   if test "$name_mangling" != "0u" -a "$name_mangling" != "1u" -a "$name_mangling" != "2u" -a "$name_mangling" != "upcase" -a "$name_mangling" != "auto" ; then
      AC_MSG_ERROR([--with-name-mangling: Invalid argument '$name_mangling'. Valid options are: 0u, 1u, 2u, upcase, auto.])
   fi

   AC_MSG_CHECKING(for Fortran PPACX symbols name decoration scheme)

   if test "$name_mangling" != "auto" ; then
      if test "$name_mangling" = "2u" ; then
         AC_DEFINE([PPACX_DOUBLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type ppacx_routine__])
         FORTRAN_DECORATION="2 underscores"
      elif test "$name_mangling" = "1u" ; then
         AC_DEFINE([PPACX_SINGLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type ppacx_routine_])
         FORTRAN_DECORATION="1 underscore"
      elif test "$name_mangling" = "upcase" ; then
         AC_DEFINE([PPACX_UPPERCASE], 1, [Defined if name decoration scheme is of type PPACX_ROUTINE])
         FORTRAN_DECORATION="UPPER CASE"
      elif test "$name_mangling" = "0u" ; then
         AC_DEFINE([PPACX_NO_UNDERSCORES], 1, [Defined if name decoration scheme is of type ppacx_routine])
         FORTRAN_DECORATION="0 underscores"
      fi
      AC_MSG_RESULT([${FORTRAN_DECORATION}])
   else

      AC_LANG_SAVE()
      AC_LANG([C])
      AX_FLAGS_SAVE()

      CC="${PACX_HOME}/bin/pacxcc"
      LIBS="${PACX_F_LIB}"

      dnl PPACX_NO_UNDERSCORES appears twice for libraries that do not support
      dnl fortran symbols 
      for ac_cv_name_mangling in \
         PPACX_NO_UNDERSCORES \
         PPACX_SINGLE_UNDERSCORE \
         PPACX_DOUBLE_UNDERSCORE \
         PPACX_UPPERCASE \
         PPACX_NO_UNDERSCORES ;
      do
         CFLAGS="-D$ac_cv_name_mangling"
   
         AC_TRY_LINK(
            [#include <pacx.h>], 
            [
               #if defined(PMPI_NO_UNDERSCORES)
               #define MY_ROUTINE ppacx_finalize
               #elif defined(PMPI_UPPERCASE)
               #define MY_ROUTINE PPACX_FINALIZE
               #elif defined(PMPI_SINGLE_UNDERSCORE)
               #define MY_ROUTINE ppacx_finalize_
               #elif defined(PMPI_DOUBLE_UNDERSCORE)
               #define MY_ROUTINE ppacx_finalize__
               #endif
   
               int ierror;
               MY_ROUTINE (&ierror);
            ],
            [
               break 
            ]
         )
      done

      AX_FLAGS_RESTORE()
      AC_LANG_RESTORE()

      if test "$ac_cv_name_mangling" = "PPACX_DOUBLE_UNDERSCORE" ; then
         AC_DEFINE([PPACX_DOUBLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type ppacx_routine__])
         FORTRAN_DECORATION="2 underscores"
      elif test "$ac_cv_name_mangling" = "PPACX_SINGLE_UNDERSCORE" ; then
         AC_DEFINE([PPACX_SINGLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type ppacx_routine_])
         FORTRAN_DECORATION="1 underscore"
      elif test "$ac_cv_name_mangling" = "PPACX_UPPERCASE" ; then
         AC_DEFINE([PPACX_UPPERCASE], 1, [Defined if name decoration scheme is of type PPACX_ROUTINE])
         FORTRAN_DECORATION="UPPER CASE"
      elif test "$ac_cv_name_mangling" = "PPACX_NO_UNDERSCORES" ; then
         AC_DEFINE([PPACX_NO_UNDERSCORES], 1, [Defined if name decoration scheme is of type ppacx_routine])
         FORTRAN_DECORATION="0 underscores"
      else
         FORTRAN_DECORATION="[unknown]"
         AC_MSG_NOTICE([Can not determine the name decoration scheme for external Fortran symbols in PACX library])
         AC_MSG_ERROR([Please use '--with-pacx-name-mangling' to select an appropriate decoration scheme.])
      fi
      AC_MSG_RESULT([${FORTRAN_DECORATION}])
   fi
])

# AX_PACX_SHOW_CONFIGURATION
# ----------
AC_DEFUN([AX_PACX_SHOW_CONFIGURATION],
[
	echo PACX instrumentation: ${PACX_INSTALLED}
	if test "${PACX_INSTALLED}" = "yes" ; then
		echo -e \\\tPACX home:          ${PACX_HOME}
	fi
])
