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
      [openshmem_paths="no"]
   )

   AC_ARG_WITH(openshmem-deps-libsdir,
      AC_HELP_STRING(
         [--with-openshmem-deps-libsdir<:@=DIR@:>@],
         [specify the directories where dependant libs are installed (e.g. -L/path/to/GASNet/lib)] 
      ),
      [openshmem_deps_libsdir=${withval}],
      [openshmem_deps_libsdir=""]
   )

   AC_ARG_WITH(openshmem-deps-libs,
      AC_HELP_STRING(
         [--with-openshmem-deps-libs<:@=LIBS@:>@],
         [specify OpenSHMEM dependant libraries (e.g. "-lgasnet-mpi-par -lammpi")] 
      ),
      [openshmem_deps_libs=${withval}],
      [openshmem_deps_libs=""]
   )
   
   if test "$openshmem_paths" != "no" ; then
      dnl Search for OpenSHMEM installation
      AX_FIND_INSTALLATION([OPENSHMEM], [$openshmem_paths], [openshmem])

      if test "${OPENSHMEM_INSTALLED}" = "yes" ; then

         OPENSHMEM_CFLAGS="-I$OPENSHMEM_INCLUDES"
         OPENSHMEM_LDFLAGS="-L$OPENSHMEM_LIBSDIR -lopenshmem ${openshmem_deps_libsdir} ${openshmem_deps_libs}"
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
      else
         AC_MSG_ERROR([Cannot find the OpenSHMEM installation. Check that --with-openshmem points to the appropriate OpenSHMEM directory.])
      fi
   fi 
   AX_FLAGS_RESTORE()

   AM_CONDITIONAL(WANT_OPENSHMEM, test "${OPENSHMEM_INSTALLED}" = "yes" -a "${openshmem_lib_works}" = "yes" )
])

# AX_OPENSHMEM_SHOW_CONFIGURATION
# -------------------------------
AC_DEFUN([AX_OPENSHMEM_SHOW_CONFIGURATION],
[
  if test "${openshmem_lib_works}" = "yes" ; then
    echo OpenSHMEM instrumentation: yes
    echo -e \\\tOpenSHMEM home:            ${OPENSHMEM_HOME}
  else
    echo OpenSHMEM instrumentation: no
  fi
])

