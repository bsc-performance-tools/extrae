# AX_ENSURE_CXX_PRESENT
# ---------------------
# Check the cxx compiler is present
AC_DEFUN([AX_ENSURE_CXX_PRESENT],
[
  AC_REQUIRE([AX_SELECT_BINARY_TYPE])

  if test "$CXX_PRESENT" != "yes" ; then
    AC_MSG_ERROR([You have enabled '$1' support which requires a working CXX compiler installed in the system, but it seems the compiler is not present. Check the 'config.log' file for previous errors or disable this option.])
  fi
])


# AX_PROG_BOOST
# --------------------
AC_DEFUN([AX_PROG_BOOST],
[
  AX_FLAGS_SAVE()

  AC_ARG_WITH(boost,
    AC_HELP_STRING(
      [--with-boost=@<:@=DIR@:>@],
      [specify where to find Boost C++ libraries and includes]
    ),
    [boost_paths="${withval}"],
    [boost_paths="no"]
  )

  if test "${boost_paths}" != "no" ; then
    AC_LANG_SAVE()
    AC_LANG_PUSH([C++])

    # Search for Boost installation
    AX_FIND_INSTALLATION([BOOST], [${boost_paths}], [], [], [boost/version.hpp], [], [], [], [], [])

    AC_LANG_RESTORE()

    if test "${BOOST_INSTALLED}" != "yes" ; then
      AC_MSG_ERROR([Boost C++ installation cannot be found. Please review --with-boost option.])
    fi
  fi

  dnl Did the checks pass?
  AM_CONDITIONAL(HAVE_BOOST, test "x${BOOST_INSTALLED}" = "xyes")

  AX_FLAGS_RESTORE()
])


# AX_PROG_ELFUTILS
# ----------------
AC_DEFUN([AX_PROG_ELFUTILS],
[
  AX_FLAGS_SAVE()

  AC_ARG_WITH(elfutils,
    AC_HELP_STRING(
      [--with-elfutils=@<:@=DIR@:>@],
      [specify where to find elfutils libraries and includes]
    ),
    [elfutils_paths="${withval}"],
    [elfutils_paths="no"]
  )

  if test "${elfutils_paths}" != "no" ; then

    # Search for elfutils installation
    AX_FIND_INSTALLATION([ELFUTILS], [${elfutils_paths}], [], [], [libelf.h dwarf.h], [], [elf dw], [], [], [])

    if test "${ELFUTILS_INSTALLED}" != "yes" ; then
      AC_MSG_ERROR([elfutils installation cannot be found. Please review --with-elfutils option.])
    fi
  fi

  dnl Did the checks pass?
  AM_CONDITIONAL(HAVE_ELFUTILS, test "x${ELFUTILS_INSTALLED}" = "xyes")

  AX_FLAGS_RESTORE()
])


# AX_PROG_TBB
# -----------
AC_DEFUN([AX_PROG_TBB],
[
  AX_FLAGS_SAVE()

  AC_ARG_WITH(tbb,
    AC_HELP_STRING(
      [--with-tbb=@<:@=DIR@:>@],
      [specify where to find Intel Threading Building Blocks libraries and includes]
    ),
    [tbb_paths="${withval}"],
    [tbb_paths="no"]
  )

  if test "${tbb_paths}" != "no" ; then
    AC_LANG_SAVE()
    AC_LANG_PUSH([C++])

    # Search for TBB installation
    AX_FIND_INSTALLATION([TBB], [${tbb_paths}], [], [], [tbb/tbb.h], [], [tbb tbbmalloc], [], [], [])

    AC_LANG_RESTORE()

    if test "${TBB_INSTALLED}" != "yes" ; then
      AC_MSG_ERROR([TBB installation cannot be found. Please review --with-tbb option.])
    fi
  fi

  dnl Did the checks pass?
  AM_CONDITIONAL(HAVE_TBB, test "x${TBB_INSTALLED}" = "xyes")

  AX_FLAGS_RESTORE()
])

# AX_PROG_DYNINST
# ---------------
AC_DEFUN([AX_PROG_DYNINST],
[
  AC_REQUIRE([AX_PROG_ELFUTILS])
  AC_REQUIRE([AX_PROG_TBB])
  AC_REQUIRE([AX_PROG_BOOST])
  AC_REQUIRE([AX_PROG_XML2([2.5.0])])

  AX_FLAGS_SAVE()

  AC_ARG_WITH(dyninst,
    AC_HELP_STRING(
      [--with-dyninst@<:@=DIR@:>@],
      [specify where to find Dyninst libraries and includes]
    ),
    [dyninst_paths="${withval}"],
    [dyninst_paths="no"]
  )

  if test "${dyninst_paths}" != "no" ; then

    dnl Check whether the required dependencies have been specified
    AX_ENSURE_CXX_PRESENT([dyninst])
    
    if test "${ELFUTILS_INSTALLED}" != "yes" ; then
      AC_MSG_ERROR([Dyninst support was activated but a required dependency was not found. Please specify where to find elfutils libraries and includes with --with-elfutils.])
    fi

    if test "${TBB_INSTALLED}" != "yes" ; then
      AC_MSG_ERROR([Dyninst support was activated but a required dependency was not found. Please specify where to find Intel TBB libraries and includes with --with-tbb.])
    fi

    if test "${BOOST_INSTALLED}" != "yes" ; then
      AC_MSG_ERROR([Dyninst support was activated but a required dependency was not found. Please specify where to find Boost C++ libraries and includes with --with-boost.])
    fi

    if test "${XML2_INSTALLED}" != "yes" ; then
      AC_MSG_ERROR([Dyninst support was activated but a required dependency was not found. Please specify where to find Boost C++ libraries and includes with --with-xml2.])
    fi

    dnl Search for Dyninst installation
    AC_LANG_SAVE()
    AC_LANG_PUSH([C++])
    dnl Dyninst >= 9.3.x requires c++11
    CXXFLAGS="-std=c++11 -I${TBB_INCLUDES} -I${BOOST_HOME}/include"
    CPPFLAGS="${CXXFLAGS}"
    AX_FIND_INSTALLATION([DYNINST], [${dyninst_paths}], [], [], [BPatch.h], [], [dyninstAPI dyninstAPI_RT instructionAPI], [], [], [])
    AC_LANG_RESTORE()

    if test "${DYNINST_INSTALLED}" = "yes" ; then
      DYNINST_CXXFLAGS="${CXXFLAGS} ${DYNINST_CXXFLAGS}"
      DYNINST_CPPFLAGS="${CPPFLAGS} ${DYNINST_CPPFLAGS}"
      DYNINST_LDFLAGS="${DYNINST_LDFLAGS} ${ELFUTILS_LDFLAGS} ${TBB_LDFLAGS}"
      DYNINST_RPATH="${DYNINST_RPATH} ${ELFUTILS_RPATH} ${TBB_RPATH}"
      AC_SUBST(DYNINST_CXXFLAGS)
      AC_SUBST(DYNINST_CPPFLAGS)
      AC_SUBST(DYNINST_LDFLAGS)
      AC_SUBST(DYNINST_RPATH)
      DYNINST_RT_LIB="${DYNINST_LIBSDIR}/libdyninstAPI_RT.so"
      AC_DEFINE_UNQUOTED([DYNINST_RT_LIB], "${DYNINST_RT_LIB}", [Define to Dyninst's library libdyninstAPI_RT.so])
    else
      AC_MSG_ERROR([Dyninst installation cannot be found. Please review --with-dyninst option.])
    fi
  fi

  dnl Did the checks pass?
  AM_CONDITIONAL(HAVE_DYNINST, test "${DYNINST_INSTALLED}" = "yes")

  AX_FLAGS_RESTORE()
])


