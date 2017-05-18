# AX_PROG_MPI
# -----------
AC_DEFUN([AX_PROG_MPI],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(mpi-lib-name,
      AC_HELP_STRING(
         [--with-mpi-lib-name@<:@=library name@:>@],
         [specify the name of the MPI library (i.e. mpi, mpich, mpi_mt...)]
      ),
      [mpi_lib_name=${withval}],
      [mpi_lib_name="not_set"]
   )

   AC_ARG_WITH(mpi,
      AC_HELP_STRING(
         [--with-mpi@<:@=DIR@:>@],
         [specify where to find MPI libraries and includes]
      ),
      [mpi_paths=${withval}],
      [mpi_paths="not_set"]
   )

   if test "${mpi_paths}" = "not_set" ; then
      AC_MSG_ERROR([Attention! You have not passed the location of the MPI library through --with-mpi option. Please, use --with-mpi to specify the location of the MPI installation on your system, or if you don't want that Extrae supports MPI instrumentation use --without-mpi instead.])
   fi

   if test "${mpi_paths}" != "no"; then
      if test -z "${mpi_paths}" ; then
         AC_MSG_ERROR([Attention! You have passed an invalid MPI location.])
      fi
      if test ! -d ${mpi_paths} ; then
         AC_MSG_ERROR([Attention! You have passed an invalid MPI location.])
      fi
   fi

   dnl Search for MPI installation
   AX_FIND_INSTALLATION([MPI], [$mpi_paths], [mpi])

   if test "${MPI_INSTALLED}" = "yes" ; then

      if test -d "${MPI_INCLUDES}/mpi" ; then
         MPI_INCLUDES="${MPI_INCLUDES}/mpi"
         MPI_CFLAGS="-I${MPI_INCLUDES}"
         CPPFLAGS="${MPI_CFLAGS} ${CPPFLAGS}"
      fi

      dnl This check is for POE over linux -- libraries are installed in /opt/ibmhpc/ppe.poe/lib/libmpi{64}/libmpi_ibm.so
      if test -d "${MPI_LIBSDIR}/libmpi${BITS}" ; then
         if test -f "${MPI_LIBSDIR}/libmpi${BITS}/libmpi_ibm.so" ; then
            MPI_LIBSDIR=${MPI_LIBSDIR}/libmpi${BITS}
         fi
      elif test -d "${MPI_LIBSDIR}/libmpi" ; then
         if test -f "${MPI_LIBSDIR}/libmpi/libmpi_ibm.so" ; then
            MPI_LIBSDIR=${MPI_LIBSDIR}/libmpi
         fi
      fi

      dnl Check for the MPI header files.
      AC_CHECK_HEADERS([mpi.h], [], [MPI_INSTALLED="no"])

      if test ${MPI_INSTALLED} = "no" ; then
         AC_MSG_ERROR([Cannot find mpi.h file in the MPI specified path])
      fi

      dnl In MN, MPICH2 requires special libraries when building mpimpi2prv
      AX_CHECK_DEFINED([mpi.h], [MPICH2], [MPI_IS_MPICH2="yes"], [MPI_IS_MPICH2="no"])

      dnl This is no longer needed in MN3
      dnl if test "${MPI_IS_MPICH2}" = "yes" -a  "${IS_MN_MACHINE}" = "yes" ; then
      dnl   MPIMPI2PRV_EXTRA_LIBS="-lpmi"
      dnl  AC_SUBST(MPIMPI2PRV_EXTRA_LIBS)
      dnl fi

      dnl Check for the MPI library.
      dnl We won't use neither AC_CHECK_LIB nor AC_TRY_LINK because this library may have unresolved references to other libs (i.e: libgm).
      AC_MSG_CHECKING([for MPI library])

		if test -f "${MPI_LIBSDIR_MULTIARCH}/libmpi.a" -o \
                -f "${MPI_LIBSDIR_MULTIARCH}/libmpi.so" ; then
			MPI_LIBSDIR="${MPI_LIBSDIR_MULTIARCH}"
		fi
		
      if test "${mpi_lib_name}" != "not_set"; then
         MPI_LIBS="-l${mpi_lib_name}"
      else
         if test -f "${MPI_LIBSDIR}/libmpi.a" ; then
            if test "${OperatingSystem}" = "aix" -a "${BITS}" = "64" ; then
               MPI_LIBS="-lmpi_r"
            else
               MPI_LIBS="-lmpi"
            fi
         elif test -f "${MPI_LIBSDIR}/libmpi.so" ; then
            MPI_LIBS="-lmpi"
         elif test -f "${MPI_LIBSDIR}/libmpich-gcc.legacy.a"; then
            MPI_LIBS="-lmpich-gcc.legacy"
         elif test -f "${MPI_LIBSDIR}/libmpich-xl.legacy.a"; then
            MPI_LIBS="-lmpich-xl.legacy"
         elif test -f "${MPI_LIBSDIR}/libmpich_intel.so"; then
            MPI_LIBS="-lmpich_intel"
         elif test -f "${MPI_LIBSDIR}/libmpich.a" -o -f "${MPI_LIBSDIR}/libmpich.so" -o -f "${MPI_LIBSDIR}/shared/libmpich.so" ; then
            MPI_LIBS="-lmpich"
         elif test -f "${MPI_LIBSDIR}/libmpi_ibm.so" ; then
            MPI_LIBS="-lmpi_ibm"
         dnl Specific for BG/P machine
         elif test -f "${MPI_LIBSDIR}/libmpich.cnk.a" ; then
            MPI_LIBS="-lmpich.cnk"
         else
            MPI_LIBS="not found"
         fi
      fi
      AC_MSG_RESULT([${MPI_LIBSDIR}, ${MPI_LIBS}])

      if test "${MPI_LIBS}" = "not found" ; then
         AC_MSG_ERROR([Cannot find MPI libraries file in the MPI specified path])
      fi

      AC_MSG_CHECKING([for shared MPI library])
      if test -f "${MPI_LIBSDIR}/libmpi.so" -o -f "${MPI_LIBSDIR}/libmpich.so" -o \
         -f "${MPI_LIBSDIR}/shared/libmpi.so" -o -f "${MPI_LIBSDIR}/shared/libmpich.so" -o \
         -f "${MPI_LIBSDIR}/libmpi_ibm.so" -o -f "${MPI_LIBSDIR}/libmpich.cnk.so" -o \
         -f "${MPI_LIBSDIR}/libmpich_intel.so" -o \
         -f "${MPI_LIBSDIR}/libmpich-gcc.legacy.so" -o -f "${MPI_LIBSDIR}/libmpich-xl.legacy.so"; then
         MPI_SHARED_LIB_FOUND="yes"
      else
         MPI_SHARED_LIB_FOUND="not found"
      fi
      AC_MSG_RESULT([${MPI_SHARED_LIB_FOUND}])

      if test "${MPI_LIBSDIR}" = "not found" ; then
         MPI_INSTALLED="no"
      else
         MPI_LDFLAGS="${MPI_LDFLAGS}"
         AC_SUBST(MPI_LDFLAGS)
         AC_SUBST(MPI_LIBS)
      fi

      AC_MSG_CHECKING([for fortran MPI library])
      if test -f "${MPI_LIBSDIR}/libmpi_f77.a" -o -f "${MPI_LIBSDIR}/libmpi_f77.so" -o -f "${MPI_LIBSDIR}/shared/libmpi_f77.so" ; then
         MPI_F_LIB_FOUND="yes"
         MPI_F_LIB="-lmpi_f77"
      elif test -f "${MPI_LIBSDIR}/libfmpich.a" -o -f "${MPI_LIBSDIR}/libfmpich.so" -o -f "${MPI_LIBSDIR}/shared/libfmpich.so"; then
         MPI_F_LIB_FOUND="yes"
         MPI_F_LIB="-lfmpich"
      elif test -f "${MPI_LIBSDIR}/libmpif.a" -o -f "${MPI_LIBSDIR}/libmpif.so" -o -f "${MPI_LIBSDIR}/shared/libmpif.so"; then
         MPI_F_LIB_FOUND="yes"
         MPI_F_LIB="-lmpif"
      elif test -f "${MPI_LIBSDIR}/libmpifort.a" -o -f "${MPI_LIBSDIR}/libmpifort.so" -o -f "${MPI_LIBSDIR}/shared/libmpifort.so"; then
         MPI_F_LIB_FOUND="yes"
         MPI_F_LIB="-lmpifort"
      elif test -f "${MPI_LIBSDIR}/libmpichf77-gcc.legacy.a" ; then
         MPI_F_LIB_FOUND="yes"
         MPI_F_LIB="-lmpif77-gcc.legacy"
      elif test -f "${MPI_LIBSDIR}/libmpichf77-xl.legacy.a" ; then
         MPI_F_LIB_FOUND="yes"
         MPI_F_LIB="-lmpif77-xl.legacy"
      else
         MPI_F_LIB_FOUND="not found"
         MPI_F_LIB=""
      fi
      AC_MSG_RESULT([${MPI_F_LIB_FOUND}, ${MPI_F_LIB}])
      AC_SUBST(MPI_F_LIB)

      if test "${MPI_LIBSDIR}" = "not found" ; then
         MPI_INSTALLED="no"
      else
         MPI_LDFLAGS="${MPI_LDFLAGS}"
         AC_SUBST(MPI_LDFLAGS)
         AC_SUBST(MPI_LIBS)
      fi

      dnl If $MPICC is not set, check for mpicc under $MPI_HOME/bin. We don't want to mix multiple MPI installations.
      AC_MSG_CHECKING([for MPI C compiler])
      if test "${MPICC}" != "" ; then
         IFS=' ' read -ra MPICC_parts <<< "${MPICC}"
         if test -x ${MPICC_parts[0]} ; then
            MPICC_COMPILER=${MPICC}
            AC_MSG_RESULT([${MPICC_COMPILER}])
         else
            AC_MSG_ERROR([Cannot find MPI C compiler ${MPICC}])
         fi
      else
         mpicc_compilers="mpicc mpiicc hcc mpxlc_r mpxlc mpcc mpcc_r cmpicc mpifccpx"
         for mpicc in [$mpicc_compilers]; do
            if test -x "${MPI_HOME}/bin${BITS}/${mpicc}" ; then
               MPICC_COMPILER="${MPI_HOME}/bin${BITS}/${mpicc}"
               AC_MSG_RESULT([${MPICC_COMPILER}])
               break
            elif test -x "${MPI_HOME}/bin/${mpicc}" ; then
               MPICC_COMPILER="${MPI_HOME}/bin/${mpicc}"
               AC_MSG_RESULT([${MPICC_COMPILER}])
               break
            fi
         done
         if test "${MPICC_COMPILER}" = "" ; then
            AC_MSG_ERROR([Cannot find \${MPI_HOME}/bin/mpicc. You can pass an alternate through MPICC environment variable.])
         fi
      fi
      MPICC=${MPICC_COMPILER}
   fi

   dnl check for mpif77 under $MPI_HOME/bin
   AC_MSG_CHECKING([for MPI F77 compiler])
   if test "${MPIF77}" = "" ; then
      mpif77_compilers="mpif77 mpxlf_r mpxlf"
      for mpif77 in [$mpif77_compilers]; do
         if test -x "${MPI_HOME}/bin${BITS}/${mpif77}" ; then
            MPIF77="${MPI_HOME}/bin${BITS}/${mpif77}"
            AC_MSG_RESULT([${MPIF77}])
            break
         elif test -x "${MPI_HOME}/bin/${mpif77}" ; then
            MPIF77="${MPI_HOME}/bin/${mpif77}"
            AC_MSG_RESULT([${MPIF77}])
            break
         fi
      done
      if test "${MPIF77}" = "" ; then
         AC_MSG_RESULT([not found])
         AC_MSG_NOTICE([Cannot find \${MPI_HOME}/bin/mpif77 -or similar- using \${CC} instead])
         MPIF77_DOES_NOT_EXIST="yes"
         MPIF77=${CC}
      else
         MPIF77_DOES_NOT_EXIST="no"
      fi
   else
      if test -x ${MPIF77} ; then
         AC_MSG_RESULT([${MPIF77}])
      else
         if test -x `which ${MPIF77}` ; then
         	AC_MSG_RESULT([${MPIF77}])
         else
         	AC_MSG_ERROR([Cannot find given \${MPIF77}. Please give the full path for the MPI F77 compiler])
         fi
      fi
   fi

   dnl check for mpif90 under $MPI_HOME/bin
   AC_MSG_CHECKING([for MPI F90 compiler])
   if test "${MPIF90}" = "" ; then
      mpif90_compilers="mpif90 mpxlf_r mpxlf"
      for mpif90 in [$mpif90_compilers]; do
         if test -x "${MPI_HOME}/bin${BITS}/${mpif90}" ; then
            MPIF90="${MPI_HOME}/bin${BITS}/${mpif90}"
            AC_MSG_RESULT([${MPIF90}])
            break
         elif test -x "${MPI_HOME}/bin/${mpif90}" ; then
            MPIF90="${MPI_HOME}/bin/${mpif90}"
            AC_MSG_RESULT([${MPIF90}])
            break
         fi
      done
      if test "${MPIF90}" = "" ; then
         AC_MSG_RESULT([not found])
         AC_MSG_NOTICE([Cannot find \${MPI_HOME}/bin/mpif90 -or similar- using \${CC} instead])
         MPIF90_DOES_NOT_EXIST="yes"
         MPIF90=${CC}
      else
         MPIF90_DOES_NOT_EXIST="no"
      fi
   else
      if test -x ${MPIF90} ; then
         AC_MSG_RESULT([${MPIF90}])
      else
         if test -x `which ${MPIF90}` ; then
         	AC_MSG_RESULT([${MPIF90}])
         else
         	AC_MSG_ERROR([Cannot find given \${MPIF90}. Please give the full path for the MPI F90 compiler])
         fi
      fi
   fi

	dnl check for mpif90 under $MPI_HOME/bin
	AC_MSG_CHECKING([for MPI launcher])
	MPIRUN=""
	for mpix in [ "mpirun" "mpiexec" ]; do
		if test -x "${MPI_HOME}/bin${BITS}/${mpix}" ; then
			MPIRUN="${MPI_HOME}/bin${BITS}/${mpix}"
			break
		elif test -x "${MPI_HOME}/bin/${mpix}" ; then
			MPIRUN="${MPI_HOME}/bin/${mpix}"
			break
		fi
	done
	if test "${MPIRUN}" != "" ; then
		AC_MSG_RESULT([${MPIRUN}])
	else
		AC_MSG_RESULT([not found! -- It is not needed to compile Extrae but it is necessary to execute regression tests])
	fi

   dnl AC_SUBST(MPICC)
   AC_ARG_VAR([MPICC],[Alternate MPI C compiler - use if the MPI C compiler in the MPI installation should not be used])
   AC_SUBST(MPIF77)
   AC_SUBST(MPIF90)
   AC_SUBST(MPIRUN)

   dnl If the system do not have MPICC (or similar) be sure to add -lmpi and -Impi
   AM_CONDITIONAL(NEED_MPI_LIB_INCLUDE, test "${CC}" = "${MPICC}" )

   dnl Did the checks pass?
   AM_CONDITIONAL(HAVE_MPI, test "${MPI_INSTALLED}" = "yes")

   dnl If the system has MPI & shared libraries
   AM_CONDITIONAL(HAVE_MPI_WITH_SHARED_LIBS, test "${MPI_INSTALLED}" = "yes" -a "${MPI_SHARED_LIB_FOUND}" = "yes")

   dnl If we have detected the MPI launcher
   AM_CONDITIONAL(HAVE_MPIRUN, test "${MPIRUN}" != "")

   if test "${MPI_INSTALLED}" = "yes" ; then
      AC_DEFINE([HAVE_MPI], 1, [Determine if MPI in installed])
   fi

   AX_FLAGS_RESTORE()
])

# AX_CHECK_MPI3_VERSION
# ---------------------
AC_DEFUN([AX_CHECK_MPI3_VERSION],
[
   AC_MSG_CHECKING([if given MPI implementation supports MPI3 standard])
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${MPI_INCLUDES}"
   AC_LANG_SAVE()
   AC_LANG([C])
   AC_TRY_COMPILE(
      [#include <mpi.h>],
      [
         #if MPI_VERSION >= 3
            return 1;
         #else
             #error "MPI version different to 3"
         #endif
      ],
      [mpi3_version="yes"],
      [mpi3_version="no"]
   )
   if test "${mpi3_version}" = "yes"; then
       AC_DEFINE([MPI3], [1], [Define if we are using a MPI3 implementation])
   fi
   AX_FLAGS_RESTORE()
   AC_LANG_RESTORE()
   AC_MSG_RESULT([${mpi3_version}])
   AX_FLAGS_RESTORE()
])



# AX_CHECK_MPI_F_STATUS_IGNORE
# ---------------------
AC_DEFUN([AX_CHECK_MPI_F_STATUS_IGNORE],
[
   AC_MSG_CHECKING([if MPI_F_STATUS_IGNORE and MPI_F_STATUSES_IGNORE exist])
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${MPI_INCLUDES}"
   AC_LANG_SAVE()
   AC_LANG([C])
   AC_TRY_COMPILE(
      [#include <mpi.h>],
      [
         MPI_Status *s1 = MPI_F_STATUS_IGNORE;
         MPI_Status *s2 = MPI_F_STATUSES_IGNORE;
         return 0;
      ],
      [mpi_f_status_ignore_exists="yes"],
      [mpi_f_status_ignore_exists="no"]
   )
   AX_FLAGS_RESTORE()
   AC_LANG_RESTORE()
   AC_MSG_RESULT([${mpi_f_status_ignore_exists}])
   if test "${mpi_f_status_ignore_exists}" = "yes"; then
      AC_DEFINE_UNQUOTED([MPI_HAS_MPI_F_STATUS_IGNORE], 1, [Does the MPI_F_STATUS_IGNORE exist in the given MPI implementation?])
   fi
   AX_FLAGS_RESTORE()
])

# AX_CHECK_MPI_STATUS_SIZE
# ---------------------
AC_DEFUN([AX_CHECK_MPI_STATUS_SIZE],
[
   AC_MSG_CHECKING([for size of the MPI_Status struct])
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${MPI_INCLUDES}"

   if test "${IS_MIC_MACHINE}" = "yes" ; then
     SIZEOF_MPI_STATUS=5
   elif test "${IS_SPARC64_MACHINE}" = "yes" ; then
     SIZEOF_MPI_STATUS=5
   else
   AC_TRY_RUN(
      [
         #include <mpi.h>
         int main()
         {
            return sizeof(MPI_Status)/sizeof(int);
         }
      ],
      [ SIZEOF_MPI_STATUS="0" ],
      [ SIZEOF_MPI_STATUS="$?"]
   )
   fi
   AC_MSG_RESULT([${SIZEOF_MPI_STATUS}])
   AC_DEFINE_UNQUOTED([SIZEOF_MPI_STATUS], ${SIZEOF_MPI_STATUS}, [Size of the MPI_Status structure in "sizeof-int" terms])
   AX_FLAGS_RESTORE()
])

# AX_CHECK_MPI_SOURCE_OFFSET
#------------------------
AC_DEFUN([AX_CHECK_MPI_SOURCE_OFFSET],
[
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${MPI_INCLUDES}"

   AC_CHECK_MEMBER(MPI_Status.MPI_SOURCE,,
                [AC_MSG_ERROR([We need MPI_Status.MPI_SOURCE!])],
                [#include <mpi.h>])

   AC_MSG_CHECKING([for offset of SOURCE field in MPI_Status])
   if test "${IS_MIC_MACHINE}" = "yes" ; then
     MPI_SOURCE_OFFSET=2
   elif test "${IS_SPARC64_MACHINE}" = "yes" ; then
     MPI_SOURCE_OFFSET=0
   else
   AC_TRY_RUN(
      [
         #include <mpi.h>
         int main()
         {
            MPI_Status s;
            long addr1 = (long) &s;
            long addr2 = (long) &(s.MPI_SOURCE);

            return (addr2 - addr1)/sizeof(int);
         }
      ],
      [ MPI_SOURCE_OFFSET="0" ],
      [ MPI_SOURCE_OFFSET="$?"]
   )
   fi
   AC_MSG_RESULT([${MPI_SOURCE_OFFSET}])
   AC_DEFINE_UNQUOTED([MPI_SOURCE_OFFSET], ${MPI_SOURCE_OFFSET}, [Offset of the SOURCE field in MPI_Status in sizeof-int terms])
   AX_FLAGS_RESTORE()
])

# AX_CHECK_MPI_TAG_OFFSET
#------------------------
AC_DEFUN([AX_CHECK_MPI_TAG_OFFSET],
[
   AX_FLAGS_SAVE()
   CFLAGS="${CFLAGS} -I${MPI_INCLUDES}"

   AC_CHECK_MEMBER(MPI_Status.MPI_TAG,,
                [AC_MSG_ERROR([We need MPI_Status.MPI_TAG!])],
                [#include <mpi.h>])

   AC_MSG_CHECKING([for offset of TAG field in MPI_Status])
   if test "${IS_MIC_MACHINE}" = "yes" ; then
     MPI_TAG_OFFSET=3
   elif test "${IS_SPARC64_MACHINE}" = "yes" ; then
     MPI_TAG_OFFSET=1
   else
   AC_TRY_RUN(
      [
         #include <mpi.h>
         int main()
         {
            MPI_Status s;
            long addr1 = (long) &s;
            long addr2 = (long) &(s.MPI_TAG);

            return (addr2 - addr1)/sizeof(int);
         }
      ],
      [ MPI_TAG_OFFSET="0" ],
      [ MPI_TAG_OFFSET="$?"]
   )
   fi
   AC_MSG_RESULT([${MPI_TAG_OFFSET}])
   AC_DEFINE_UNQUOTED([MPI_TAG_OFFSET], ${MPI_TAG_OFFSET}, [Offset of the TAG field in MPI_Status in sizeof-int terms])
   AX_FLAGS_RESTORE()
])

# AX_CHECK_PMPI_NAME_MANGLING
# ---------------------------
AC_DEFUN([AX_CHECK_PMPI_NAME_MANGLING],
[
   AC_REQUIRE([AX_PROG_MPI])

   AC_ARG_WITH(mpi-name-mangling,
      AC_HELP_STRING(
         [--with-mpi-name-mangling@<:@=ARG@:>@], 
         [choose the name decoration scheme for external Fortran symbols in MPI library from: 0u, 1u, 2u, upcase, auto @<:@default=auto@:>@]
      ),
      [name_mangling="$withval"],
      [name_mangling="auto"]
   )

   if test "$name_mangling" != "0u" -a \
           "$name_mangling" != "1u" -a \
           "$name_mangling" != "2u" -a \
           "$name_mangling" != "_f" -a \
           "$name_mangling" != "upcase" -a \
           "$name_mangling" != "auto" ; then
      AC_MSG_ERROR([--with-name-mangling: Invalid argument '$name_mangling'. Valid options are: 0u, 1u, 2u, _f, upcase, auto.])
   fi

   AC_MSG_CHECKING(for Fortran PMPI symbols name decoration scheme)

   if test "$name_mangling" != "auto" ; then
      if test "$name_mangling" = "2u" ; then
         AC_DEFINE([PMPI_DOUBLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type pmpi_routine__])
         FORTRAN_DECORATION="2 underscores"
      elif test "$name_mangling" = "1u" ; then
         AC_DEFINE([PMPI_SINGLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type pmpi_routine_])
         FORTRAN_DECORATION="1 underscore"
      elif test "$name_mangling" = "upcase" ; then
         AC_DEFINE([PMPI_UPPERCASE], 1, [Defined if name decoration scheme is of type PMPI_ROUTINE])
         FORTRAN_DECORATION="UPPER CASE"
      elif test "$name_mangling" = "0u" ; then
         AC_DEFINE([PMPI_NO_UNDERSCORES], 1, [Defined if name decoration scheme is of type pmpi_routine])
         FORTRAN_DECORATION="0 underscores"
      elif test "$name_mangling" = "_f" ; then
         AC_DEFINE([PMPI_UNDERSCORE_F_SUFFIX], 1, [Defined if name decoration scheme is of type pmpi_routine_f])
         FORTRAN_DECORATION="_f suffix"
      fi
      AC_MSG_RESULT([${FORTRAN_DECORATION}])
   else

      AC_LANG_SAVE()
      AC_LANG([C])
      AX_FLAGS_SAVE()

      CC="${MPICC}"

      dnl PMPI_NO_UNDERSCORES appears twice for libraries that do not support
      dnl fortran symbols 
      for ac_cv_name_mangling in \
         PMPI_NO_UNDERSCORES \
         PMPI_SINGLE_UNDERSCORE \
         PMPI_DOUBLE_UNDERSCORE \
         PMPI_UPPERCASE \
         PMPI_UNDERSCORE_F_SUFFIX \
         PMPI_NO_UNDERSCORES ;
      do
         CFLAGS="-D$ac_cv_name_mangling"
         LIBS="${LIBS} ${MPI_F_LIB}" dnl We need to append fortran libraries if they exist
   
         AC_TRY_LINK(
            [#include <mpi.h>], 
            [
               #if defined(PMPI_NO_UNDERSCORES)
               #define MY_ROUTINE pmpi_finalize
               #elif defined(PMPI_UPPERCASE)
               #define MY_ROUTINE PMPI_FINALIZE
               #elif defined(PMPI_SINGLE_UNDERSCORE)
               #define MY_ROUTINE pmpi_finalize_
               #elif defined(PMPI_DOUBLE_UNDERSCORE)
               #define MY_ROUTINE pmpi_finalize__
               #elif defined(PMPI_UNDERSCORE_F_SUFFIX)
               #define MY_ROUTINE pmpi_finalize_f
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

      if test "$ac_cv_name_mangling" = "PMPI_DOUBLE_UNDERSCORE" ; then
         AC_DEFINE([PMPI_DOUBLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type pmpi_routine__])
         FORTRAN_DECORATION="2 underscores"
      elif test "$ac_cv_name_mangling" = "PMPI_SINGLE_UNDERSCORE" ; then
         AC_DEFINE([PMPI_SINGLE_UNDERSCORE], 1, [Defined if name decoration scheme is of type pmpi_routine_])
         FORTRAN_DECORATION="1 underscore"
      elif test "$ac_cv_name_mangling" = "PMPI_UPPERCASE" ; then
         AC_DEFINE([PMPI_UPPERCASE], 1, [Defined if name decoration scheme is of type PMPI_ROUTINE])
         FORTRAN_DECORATION="UPPER CASE"
      elif test "$ac_cv_name_mangling" = "PMPI_NO_UNDERSCORES" ; then
         AC_DEFINE([PMPI_NO_UNDERSCORES], 1, [Defined if name decoration scheme is of type pmpi_routine])
         FORTRAN_DECORATION="0 underscores"
      elif test "${ac_cv_name_mangling}" = "PMPI_UNDERSCORE_F_SUFFIX" ; then
         AC_DEFINE([PMPI_UNDERSCORE_F_SUFFIX], 1, [Defined if name decoration scheme is of type pmpi_routine_f])
         FORTRAN_DECORATION="_f suffix"
      else
         FORTRAN_DECORATION="[unknown]"
         AC_MSG_NOTICE([Can not determine the name decoration scheme for external Fortran symbols in MPI library])
         AC_MSG_ERROR([Please use '--with-mpi-name-mangling' to select an appropriate decoration scheme.])
      fi
      AC_MSG_RESULT([${FORTRAN_DECORATION}])
   fi
])

# AX_CHECK_MPI_SUPPORTS_MPI_COMM_SPAWN
# ---------
AC_DEFUN([AX_CHECK_MPI_SUPPORTS_MPI_COMM_SPAWN],
[
	AC_LANG_SAVE()
	AC_LANG([C])
	AX_FLAGS_SAVE()

	AC_MSG_CHECKING([if MPI library supports MPI_Comm_spawn])
	AC_TRY_LINK(
		[#include <mpi.h>], 
		[
			int ierror;
			int errcodes[1];
			MPI_Comm c;
			MPI_Info i;
			char *argv[1];
			ierror = MPI_Comm_spawn ("", argv, 1, i, 0, MPI_COMM_WORLD, &c, errcodes);
		],
		[mpi_lib_supports_mpi_comm_spawn="yes" ],
		[mpi_lib_supports_mpi_comm_spawn="no" ]
	)
	AC_MSG_RESULT([${mpi_lib_supports_mpi_comm_spawn}])

	if test "${mpi_lib_supports_mpi_comm_spawn}" = "yes" ; then
		AC_DEFINE([MPI_SUPPORTS_MPI_COMM_SPAWN], 1, [Defined if MPI library supports MPI_Comm_spawn])
	fi

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

# AX_CHECK_MPI_SUPPORTS_MPI_1SIDED
# ---------
AC_DEFUN([AX_CHECK_MPI_SUPPORTS_MPI_1SIDED],
[
	AC_LANG_SAVE()
	AC_LANG([C])
	AX_FLAGS_SAVE()

	CC="${MPICC}"

	AC_MSG_CHECKING([if MPI library supports MPI 1-sided operations])
	AC_TRY_LINK(
		[#include <mpi.h>], 
		[
			int ierror;
			ierror = MPI_Put ((void*)0, 0, (MPI_Datatype)0, 0, (MPI_Aint)0, 0, (MPI_Datatype)0, (MPI_Win)0);
			ierror = MPI_Get ((void*)0, 0, (MPI_Datatype)0, 0, (MPI_Aint)0, 0, (MPI_Datatype)0, (MPI_Win)0);
		],
		[mpi_lib_supports_mpi_1sided="yes" ],
		[mpi_lib_supports_mpi_1sided="no" ]
	)
	AC_MSG_RESULT([${mpi_lib_supports_mpi_1sided}])

	if test "${mpi_lib_supports_mpi_1sided}" = "yes" ; then
		AC_DEFINE([MPI_SUPPORTS_MPI_1SIDED], 1, [Defined if MPI library supports 1-sided operations])
	fi

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

# AX_CHECK_MPI_SUPPORTS_MPI_IO
# ---------
AC_DEFUN([AX_CHECK_MPI_SUPPORTS_MPI_IO],
[
	AC_LANG_SAVE()
	AC_LANG([C])
	AX_FLAGS_SAVE()

	CC="${MPICC}"

	AC_MSG_CHECKING([if MPI library supports MPI I/O])
	AC_TRY_LINK(
		[#include <mpi.h>], 
		[
			int ierror;
			MPI_Info i;
			MPI_File f;
			ierror = MPI_File_open (MPI_COMM_WORLD, 0, MPI_MODE_CREATE, i, &f);
		],
		[mpi_lib_supports_mpi_io="yes" ],
		[mpi_lib_supports_mpi_io="no" ]
	)
	AC_MSG_RESULT([${mpi_lib_supports_mpi_io}])

	if test "${mpi_lib_supports_mpi_io}" = "yes" ; then
		AC_DEFINE([MPI_SUPPORTS_MPI_IO], 1, [Defined if MPI library supports I/O operations])
	fi

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

# AX_CHECK_MPI_C_HAS_FORTRAN_MPI_INIT
# ---------
AC_DEFUN([AX_CHECK_MPI_C_HAS_FORTRAN_MPI_INIT],
[
	AC_LANG_SAVE()
	AC_LANG([C])
	AX_FLAGS_SAVE()

	CC="${MPICC}"

	AC_MSG_CHECKING([if MPI C library contains Fortran MPI_Init symbol])
	AC_TRY_LINK(
		[#include <mpi.h>], 
		[
			int ierror;
			ierror = mpi_init (&ierror);
		],
		[mpi_clib_contains_fortran_mpi_init="yes" ],
		[mpi_clib_contains_fortran_mpi_init="no" ]
	)
	AC_MSG_RESULT([${mpi_clib_contains_fortran_mpi_init}])

	if test "${mpi_clib_contains_fortran_mpi_init}" = "yes" ; then
		AC_DEFINE([MPI_C_CONTAINS_FORTRAN_MPI_INIT], 1, [Defined if MPI C library contains Fortran mpi_init symbol])
	fi

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

# AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD_C
# ---------
AC_DEFUN([AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD_C],
[
	AC_LANG_SAVE()
	AC_LANG([C])
	AX_FLAGS_SAVE()

	CC="${MPICC}"

	AC_MSG_CHECKING([if MPI library supports threads using MPI_Init_thread (C)])
	AC_TRY_LINK(
		[#include <mpi.h>], 
		[
				#if defined(PMPI_NO_UNDERSCORES)
				# define MY_ROUTINE mpi_init_thread
				#elif defined(PMPI_UPPERCASE)
				# define MY_ROUTINE MPI_INIT_THREAD
				#elif defined(PMPI_SINGLE_UNDERSCORE)
				# define MY_ROUTINE mpi_init_thread_
				#elif defined(PMPI_DOUBLE_UNDERSCORE)
				# define MY_ROUTINE mpi_init_thread__
				#endif
				int ierror;
				ierror = MPI_Init_thread (0, 0, MPI_THREAD_FUNNELED, 0);
		],
		[mpi_clib_contains_mpi_init_thread="yes" ],
		[mpi_clib_contains_mpi_init_thread="no" ]
	)
	AC_MSG_RESULT([${mpi_clib_contains_mpi_init_thread}])

	if test "${mpi_clib_contains_mpi_init_thread}" = "yes" ; then
		AC_DEFINE([MPI_HAS_INIT_THREAD_C], 1, [Defined if MPI library supports MPI_Init_thread / C])
	fi

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

# AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD_F
# ---------
AC_DEFUN([AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD_F],
[
	AC_LANG_SAVE()
	AC_LANG([C])
	AX_FLAGS_SAVE()

	AC_MSG_CHECKING([if MPI library supports threads using MPI_Init_thread (Fortran)])

	if test "${MPIF90_DOES_NOT_EXIST}" = "yes" ; then
		dnl If we can't compile the fortran app, just guess from C library
		mpi_flib_contains_mpi_init_thread=${mpi_clib_contains_mpi_init_thread}
		guessed=" (guessed from C library)"
	else
		FC="${MPIF90}"
		AC_LANG_PUSH([Fortran])
		AC_LINK_IFELSE(
			[        program test
	         call mpi_init_thread()
	         end],
			[mpi_flib_contains_mpi_init_thread="yes" ],
			[mpi_flib_contains_mpi_init_thread="no" ]
		)
		AC_LANG_POP([Fortran])
		guessed=""
	fi
	AC_MSG_RESULT([${mpi_flib_contains_mpi_init_thread}${guessed}])

	if test "${mpi_flib_contains_mpi_init_thread}" = "yes" ; then
		AC_DEFINE([MPI_HAS_INIT_THREAD_F], 1, [Defined if MPI library supports MPI_Init_thread / Fortran])
	fi

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

AC_DEFUN([AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD],
[
  AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD_C
  AX_CHECK_MPI_LIB_HAS_MPI_INIT_THREAD_F
])


# AX_CHECK_MPI_LIB_HAS_C_AND_FORTRAN_SYMBOLS
# ---------
AC_DEFUN([AX_CHECK_MPI_LIB_HAS_C_AND_FORTRAN_SYMBOLS],
[
	AC_LANG_SAVE()
	AC_LANG([C])
	AX_FLAGS_SAVE()

	CC="${MPICC}"

	if test ${MPI_INSTALLED} = "yes" ; then
		AC_MSG_CHECKING([if MPI library contains both C and Fortran symbols])
		AC_TRY_LINK(
			[#include <mpi.h>], 
			[
				#if defined(PMPI_NO_UNDERSCORES)
				# define MY_ROUTINE pmpi_finalize
				#elif defined(PMPI_UPPERCASE)
				# define MY_ROUTINE PMPI_FINALIZE
				#elif defined(PMPI_SINGLE_UNDERSCORE)
				# define MY_ROUTINE pmpi_finalize_
				#elif defined(PMPI_DOUBLE_UNDERSCORE)
				# define MY_ROUTINE pmpi_finalize__
				#endif

				int ierror;
				MY_ROUTINE (&ierror);
				ierror = MPI_Finalize ();
			],
			[mpi_lib_contains_c_and_fortran="yes" ],
			[mpi_lib_contains_c_and_fortran="no" ]
		)
		AC_MSG_RESULT([${mpi_lib_contains_c_and_fortran}])
	fi

	AM_CONDITIONAL(COMBINED_C_FORTRAN, test "${mpi_lib_contains_c_and_fortran}" = "yes")
	if test "${mpi_lib_contains_c_and_fortran}" = "yes" ; then
		AC_DEFINE([MPI_COMBINED_C_FORTRAN], 1, [Defined if a single MPI library contains both C and Fortran symbols])
	fi

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

# AX_ENABLE_SINGLE_MPI_LIBRARY
# ---------
AC_DEFUN([AX_ENABLE_SINGLE_MPI_LIBRARY],
[
   AC_ARG_ENABLE(single-mpi-lib,
      AC_HELP_STRING(
         [--enable-single-mpi-lib],
         [Produces a single instrumentation library for MPI that contains both Fortran and C wrappers]
      ),
      [enable_single_mpi_lib="${enableval}"],
      [enable_single_mpi_lib="yes"]
   )

   AM_CONDITIONAL(SINGLE_MPI_LIBRARY, test "${enable_single_mpi_lib}" = "yes")
])


# AX_PROG_GM
# ----------
AC_DEFUN([AX_PROG_GM],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(gm,
      AC_HELP_STRING(
         [--with-gm@<:@=DIR@:>@],
         [specify where to find GM libraries and includes]
      ),
      [gm_paths="$withval"],
      [gm_paths="/opt/osshpc/gm"] dnl List of possible default paths
   )

   dnl Search for GM installation
   AX_FIND_INSTALLATION([GM], [${gm_paths}], [gm])

   if test "$GM_INSTALLED" = "yes" ; then
      dnl Check for GM header files.
      AC_CHECK_HEADERS([gm.h], [], [GM_INSTALLED="no"])

      dnl Check for libgm
      AC_CHECK_LIB([gm], [_gm_get_globals], 
         [ 
           GM_LDFLAGS="$GM_LDFLAGS -lgm"
           AC_SUBST(GM_LDFLAGS)
         ],
         [ GM_INSTALLED="no"]
      )
   fi

   dnl Did the checks pass?
   AM_CONDITIONAL(HAVE_GM, test "${GM_INSTALLED}" = "yes")

   if test "$GM_INSTALLED" = "no" ; then
      AC_MSG_WARN([Myrinet GM counters tracing has been disabled])
   fi

   AX_FLAGS_RESTORE()
])

# AX_MPI_SHOW_CONFIGURATION
# ----------
AC_DEFUN([AX_MPI_SHOW_CONFIGURATION],
[
	echo MPI instrumentation: ${MPI_INSTALLED}
	if test "${MPI_INSTALLED}" = "yes" ; then
		MPI_EXTRA_CAPABILITIES=""
		if test "${PERUSE_AVAILABILITY}" = "yes"; then
			MPI_EXTRA_CAPABILITIES+=" peruse"
		fi
		if test "${mpi_lib_supports_mpi_1sided}" = "yes"; then
			MPI_EXTRA_CAPABILITIES+=" 1-sided"
		fi
		if test "${mpi_lib_supports_mpi_io}" = "yes"; then
			MPI_EXTRA_CAPABILITIES+=" I/O"
		fi
		if test "${mpi_lib_supports_mpi_comm_spawn}" = "yes"; then
			MPI_EXTRA_CAPABILITIES+=" MPI_Comm_spawn"
		fi
		if test "${mpi3_version}" = "yes"; then
			MPI_EXTRA_CAPABILITIES+=" MPI3"
		fi
		echo -e \\\tMPI home:             ${MPI_HOME}
		if test "${MPIRUN}" != ""; then
		echo -e \\\tMPI launcher:         ${MPIRUN}
		else
		echo -e \\\tMPI launcher was not found. It is NOT necessary to compile Extrae but it is necessary to execute the regression tests.
		fi
		echo -e \\\tFortran decoration:   ${FORTRAN_DECORATION}
		echo -e \\\tmixed C/Fortran libraries? ${mpi_lib_contains_c_and_fortran}
		echo -e \\\tshared libraries?     ${MPI_SHARED_LIB_FOUND}
		echo -e \\\tMPI capabilities:     ${MPI_EXTRA_CAPABILITIES}
	fi
])

# AX_PMPI_HOOK
# ------------
AC_DEFUN([AX_PMPI_HOOK],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH([pmpi-hook],
   	AS_HELP_STRING([--with-pmpi-hook], [Choose method to call PMPI (dlsym or pmpi)]))
   AS_IF([test "x$with_pmpi_hook" == "xdlsym"],
   	[AC_DEFINE([WITH_PMPI_HOOK], [1], [Use dlsym instead of PMPI])])

   AX_FLAGS_RESTORE()
])
