# AX_PROG_OPENSHMEM
# -----------------
AC_DEFUN([AX_PROG_OPENSHMEM],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(openshmem,
      AC_HELP_STRING(
         [--with-openshmem@<:@=DIR@:>@],
         [specify where to find OpenSHMEM libraries and includes]
      ),
      [openshmem_paths=${withval}],
      [openshmem_paths="not_set"]
   )
   
   if test "$openshmem_paths" != "not_set" ; then
      dnl Search for OpenSHMEM installation
      AX_FIND_INSTALLATION([OPENSHMEM], [$openshmem_paths], [openshmem])

      if test "${OPENSHMEM_INSTALLED}" = "yes" ; then

         OPENSHMEM_CFLAGS="-I$OPENSHMEM_INCLUDES"
         OPENSHMEM_LDFLAGS="-L$OPENSHMEM_LIBSDIR -lopenshmem -L/home/gllort/Apps/GASNet/1.22.4/lib -lgasnet-mpi-par -lammpi"
         CFLAGS="$OPENSHMEM_CFLAGS $CFLAGS"
         LDFLAGS="$OPENSHMEM_LDFLAGS $LDFLAGS"
         LIBOPENSHMEM="-lopenshmem"

         dnl Check for the OpenSHMEM header files.
         AC_CHECK_HEADERS([shmem.h], [], [OPENSHMEM_INSTALLED="no"])

         if test ${OPENSHMEM_INSTALLED} = "no" ; then
            AC_MSG_ERROR([Cannot find shmem.h file in the OpenSHMEM specified path])
         fi

         dnl Check for the OPENSHMEM library.
         AC_MSG_CHECKING([for OpenSHMEM library])

         AC_TRY_LINK(
           [ #include <shmem.h> ],
           [ return 0;          ], # XXX start_pes
           [ openshmem_lib_works="yes" ],
           [ openshmem_lib_works="no"  ]
         )

         AC_MSG_RESULT([${openshmem_lib_works}])

         if test "${openshmem_lib_works}" = "yes"; then
            AC_DEFINE([WANT_OPENSHMEM], [1], [OpenSHMEM required])
            AC_DEFINE([HAVE_SHMEM_H], [1], [Define to 1 if you have <shmem.h> header file])
            AC_SUBST(LIBOPENSHMEM)
            AC_SUBST(OPENSHMEM_LDFLAGS)
         else
            AC_MSG_ERROR([Cannot link OpenSHMEM test. Check that --with-openshmem points to the appropriate OpenSHMEM directory.])
         fi
      fi
   else
      AC_MSG_ERROR([Cannot find the OpenSHMEM installation. Check that --with-openshmem points to the appropriate OpenSHMEM directory.])
   fi 
   AX_FLAGS_RESTORE()

   AM_CONDITIONAL(WANT_OPENSHMEM, test "${OPENSHMEM_INSTALLED}" = "yes" -a "${openshmem_lib_works}" = "yes" )
])
