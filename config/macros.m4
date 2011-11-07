# AX_FLAGS_SAVE
# -------------
AC_DEFUN([AX_FLAGS_SAVE],
[
   saved_LIBS="${LIBS}"
   saved_CC="${CC}"
   saved_CFLAGS="${CFLAGS}"
   saved_CXXFLAGS="${CXXFLAGS}"
   saved_CPPFLAGS="${CPPFLAGS}"
   saved_LDFLAGS="${LDFLAGS}"
])


# AX_FLAGS_RESTORE
# ----------------
AC_DEFUN([AX_FLAGS_RESTORE],
[
   LIBS="${saved_LIBS}"
   CC="${saved_CC}"
   CFLAGS="${saved_CFLAGS}"
   CXXFLAGS="${saved_CXXFLAGS}"
   CPPFLAGS="${saved_CPPFLAGS}"
   LDFLAGS="${saved_LDFLAGS}"
])


# AX_FIND_INSTALLATION
# --------------------
AC_DEFUN([AX_FIND_INSTALLATION],
[
	AC_REQUIRE([AX_SELECT_BINARY_TYPE])

	dnl Search for home directory
	AC_MSG_CHECKING([for $1 installation])
    for home_dir in [$2 "not found"]; do
        if test -d "$home_dir/$BITS" ; then
            home_dir="$home_dir/$BITS"
            break
        elif test -d "$home_dir" ; then
            break
        fi
    done
	AC_MSG_RESULT([$home_dir])
	$1_HOME="$home_dir"
	if test "$$1_HOME" = "not found" ; then
		$1_HOME=""
	else

		dnl Did the user passed a headers directory to check first?
		AC_ARG_WITH([$3-headers],
			AC_HELP_STRING(
				[--with-$3-headers@<:@=ARG@:>@],
				[Specify location of include files for package $3]
			),
			[ForcedHeaders="$withval"],
			[ForcedHeaders=""]
		)

		dnl Search for includes directory
		AC_MSG_CHECKING([for $1 includes directory])

		if test "${ForcedHeaders}" = "" ; then
			for incs_dir in [$$1_HOME/include$BITS $$1_HOME/include "not found"] ; do
				if test -d "$incs_dir" ; then
					break
				fi
			done
		else
			for incs_dir in [${ForcedHeaders} "not found"] ; do
				if test -d "$incs_dir" ; then
					break
				fi
			done
		fi

		AC_MSG_RESULT([$incs_dir])
		$1_INCLUDES="$incs_dir"
		if test "$$1_INCLUDES" = "not found" ; then
			AC_MSG_ERROR([Unable to find header directory for package $3. Check option --with-$3-headers.])
		else
			$1_CFLAGS="-I$$1_INCLUDES"
			$1_CXXFLAGS="-I$$1_INCLUDES"
			$1_CPPFLAGS="-I$$1_INCLUDES"
		fi

		dnl Did the user passed a headers directory to check first?
		AC_ARG_WITH([$3-libs],
			AC_HELP_STRING(
				[--with-$3-libs@<:@=ARG@:>@],
				[Specify location of library files for package $3]
			),
			[ForcedLibs="$withval"],
			[ForcedLibs=""]
		)

		dnl Search for libs directory
		AC_MSG_CHECKING([for $1 libraries directory])
		if test "${ForcedLibs}" = "" ; then
			for libs_dir in [$$1_HOME/lib$BITS $$1_HOME/lib "not found"] ; do
				if test -d "$libs_dir" ; then
					break
				fi
			done
		else
			for libs_dir in [${ForcedLibs} "not found"] ; do
				if test -d "$libs_dir" ; then
					break
				fi
			done
		fi

		AC_MSG_RESULT([$libs_dir])
		$1_LIBSDIR="$libs_dir"
		if test "$$1_LIBSDIR" = "not found" ; then
			AC_MSG_ERROR([Unable to find library directory for package $3. Check option --with-$3-libs.])
		else
       $1_LDFLAGS="-L$$1_LIBSDIR"
       if test -d "$$1_LIBSDIR/shared" ; then
          $1_SHAREDLIBSDIR="$$1_LIBSDIR/shared"
       else
          $1_SHAREDLIBSDIR=$$1_LIBSDIR
       fi
		fi
	fi

	dnl Everything went OK?
	if test "$$1_HOME" != "" -a "$$1_INCLUDES" != "" -a "$$1_LIBSDIR" != "" ; then
		$1_INSTALLED="yes"

		AC_SUBST($1_HOME)
		AC_SUBST($1_INCLUDES)

    AC_SUBST($1_CFLAGS)
    AC_SUBST($1_CXXFLAGS)
    AC_SUBST($1_CPPFLAGS)

    AC_SUBST($1_LDFLAGS)
    AC_SUBST($1_SHAREDLIBSDIR)
    AC_SUBST($1_LIBSDIR)

    dnl Update the default variables so the automatic checks will take into account the new directories
    CFLAGS="$CFLAGS $$1_CFLAGS"
    CXXFLAGS="$CXXFLAGS $$1_CXXFLAGS"
    CPPFLAGS="$CPPFLAGS $$1_CPPFLAGS"
    LDFLAGS="$LDFLAGS $$1_LDFLAGS"
	else	
		$1_INSTALLED="no"
	fi
])


# AX_CHECK_POINTER_SIZE
# ---------------------
AC_DEFUN([AX_CHECK_POINTER_SIZE],
[
   AC_TRY_RUN(
      [
         int main()
         {
            return sizeof(void *)*8;
         }
      ],
      [ POINTER_SIZE="0" ],
      [ POINTER_SIZE="$?"]
   )
])


# AX_SELECT_BINARY_TYPE
# ---------------------
# Check the binary type the user wants to build and verify whether it can be successfully built
AC_DEFUN([AX_SELECT_BINARY_TYPE],
[
	AC_ARG_WITH(binary-type,
		AC_HELP_STRING(
			[--with-binary-type@<:@=ARG@:>@],
			[choose the binary type between: 32, 64, default @<:@default=default@:>@]
		),
		[Selected_Binary_Type="$withval"],
		[Selected_Binary_Type="default"]
	)

	if test "$Selected_Binary_Type" != "default" -a "$Selected_Binary_Type" != "32" -a "$Selected_Binary_Type" != "64" ; then
		AC_MSG_ERROR([--with-binary-type: Invalid argument '$Selected_Binary_Type'. Valid options are: 32, 64, default.])
	fi

	C_compiler="$CC"
	CXX_compiler="$CXX"

	AC_LANG_SAVE([])
	m4_foreach([language], [[C], [C++]], [
		AC_LANG_PUSH(language)

		AC_CACHE_CHECK(
			[for $_AC_LANG_PREFIX[]_compiler compiler default binary type], 
			[[]_AC_LANG_PREFIX[]_ac_cv_compiler_default_binary_type],
			[
				AX_CHECK_POINTER_SIZE
				Default_Binary_Type="$POINTER_SIZE"
				[]_AC_LANG_PREFIX[]_ac_cv_compiler_default_binary_type="$Default_Binary_Type""-bit"
			]
		)

		if test "$Default_Binary_Type" != "32" -a "$Default_Binary_Type" != 64 ; then
			AC_MSG_ERROR([Unknown default binary type (pointer size is $POINTER_SIZE!?)])
		fi

		if test "$Selected_Binary_Type" = "default" ; then
			Selected_Binary_Type="$Default_Binary_Type"
		fi

		if test "$Selected_Binary_Type" != "$Default_Binary_Type" ; then

			force_bit_flags="-m32 -q32 -32 -maix32 -m64 -q64 -64 -maix64 none"

			AC_MSG_CHECKING([for $_AC_LANG_PREFIX[]_compiler compiler flags to build a $Selected_Binary_Type-bit binary])
			for flag in [$force_bit_flags]; do
				old_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
				[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS $flag"

				AX_CHECK_POINTER_SIZE()
				if test "$POINTER_SIZE" = "$Selected_Binary_Type" ; then
					AC_MSG_RESULT([$flag])
					break
				else
					[]_AC_LANG_PREFIX[]FLAGS="$old_[]_AC_LANG_PREFIX[]FLAGS"
					if test "$flag" = "none" ; then
						AC_MSG_RESULT([unknown])
						AC_MSG_NOTICE([${Selected_Binary_Type}-bit binaries not supported])
						AC_MSG_ERROR([Please use '--with-binary-type' to select an appropriate binary type.])

					fi
				fi
			done
		fi
		AC_LANG_POP(language)
	])
	AC_LANG_RESTORE([])
	BITS="$Selected_Binary_Type"
])


# AX_CHECK_ENDIANNESS
# -------------------
# Test if the architecture is little or big endian
AC_DEFUN([AX_CHECK_ENDIANNESS],
[
   AC_CACHE_CHECK([for the architecture endianness], [ac_cv_endianness],
   [
      AC_LANG_SAVE()
      AC_LANG([C])
      AC_TRY_RUN(
      [
         int main()
         {
            short s = 1;
            short * ptr = &s;
            unsigned char c = *((char *)ptr);
            return c;
         }
      ],
      [ac_cv_endianness="big endian" ],
      [ac_cv_endianness="little endian" ]
      )
      AC_LANG_RESTORE()
   ])
   if test "$ac_cv_endianness" = "big endian" ; then
      AC_DEFINE(IS_BIG_ENDIAN, 1, [Define to 1 if architecture is big endian])
   fi
   if test "$ac_cv_endianness" = "little endian" ; then
      AC_DEFINE(IS_LITTLE_ENDIAN, 1, [Define to 1 if architecture is little endian])
   fi
])


# AX_CHECK__FUNCTION__MACRO
# -------------------------
# Check whether the compiler defines the __FUNCTION__ macro
AC_DEFUN([AX_CHECK__FUNCTION__MACRO],
[
   AC_CACHE_CHECK([whether the compiler defines the __FUNCTION__ macro], [ac_cv_have__function__],
      [
         AC_LANG_SAVE()
         AC_LANG([C])
         AC_TRY_COMPILE(
            [#include <stdio.h>],
            [
               char *s = __FUNCTION__;
               return 0;
            ],
            [ac_cv_have__function__="yes"],
            [ac_cv_have__function__="no"]
         )
         AC_LANG_RESTORE()
      ]
   )
   if test "$ac_cv_have__function__" = "yes" ; then
      AC_DEFINE([HAVE__FUNCTION__], 1, [Define to 1 if __FUNCTION__ macro is supported])
   fi
])

AC_DEFUN([AX_CHECK_PGI],
[
   AC_MSG_CHECKING(for PGI C compiler)
   AX_FLAGS_SAVE()
   AC_LANG_SAVE()
   AC_LANG([C])
   AC_TRY_COMPILE(
      [],
      [
         #if !defined (__PGI__) && !defined(__PGI)
         # error "This is for PGI compilers only"
         #endif
         return 0;
      ],
      [pgi_compiler="yes"],
      [pgi_compiler="no"]
   )
   AX_FLAGS_RESTORE()
   AC_LANG_RESTORE()
   if test "${pgi_compiler}" = "yes"; then
      AC_MSG_RESULT([yes])
   else
      AC_MSG_RESULT([no])
   fi
])

# AX_PROG_XML2
# -----------
AC_DEFUN([AX_PROG_XML2],
[
   XML2_HOME_BIN="`dirname ${XML2_CONFIG}`"
   XML2_HOME="`dirname ${XML2_HOME_BIN}`"

   XML2_INCLUDES1="${XML2_HOME}/include/libxml2"
   XML2_INCLUDES2="${XML2_HOME}/include"
   XML2_CFLAGS="-I${XML2_INCLUDES1} -I${XML2_INCLUDES2}"
   XML2_CPPFLAGS=${XML2_CFLAGS}
   XML2_CXXFLAGS=${XML2_CFLAGS}

   XML2_LIBS="-lxml2"
   if test -f ${XML2_HOME}/lib${BITS}/libxml2.so -o -f ${XML2_HOME}/lib${BITS}/libxml2.a ; then
      XML2_LIBSDIR="${XML2_HOME}/lib${BITS}"
   else
      XML2_LIBSDIR="${XML2_HOME}/lib"
   fi
   XML2_LDFLAGS="-L${XML2_LIBSDIR}"

   if test -d ${XML2_LIBSDIR}/shared ; then 
      XML2_SHAREDLIBSDIR="${XML2_LIBSDIR}/shared"
   else
      XML2_SHAREDLIBSDIR=${XML2_LIBSDIR}
   fi

   AC_SUBST(XML2_HOME)
   AC_SUBST(XML2_CFLAGS)
   AC_SUBST(XML2_CPPFLAGS)
   AC_SUBST(XML2_CXXFLAGS)
   AC_SUBST(XML2_INCLUDES)
   AC_SUBST(XML2_LIBSDIR)
   AC_SUBST(XML2_SHAREDLIBSDIR)
   AC_SUBST(XML2_LIBS)
   AC_SUBST(XML2_LDFLAGS)
])

# AX_PROG_BFD
# -----------
AC_DEFUN([AX_PROG_BFD],
[
   BFD_INSTALLED="no"
   LIBERTY_INSTALLED="no"

   if test "${IS_BGL_MACHINE}" = "yes" -o "${IS_BGP_MACHINE}" = "yes" ; then
      bfd_default_paths="${BG_HOME}/blrts-gnu"
      libiberty_default_paths="${BG_HOME}/blrts-gnu"
   else
      bfd_default_paths="/usr /usr/local"
      libiberty_default_paths="/usr /usr/local"
   fi

   AC_MSG_CHECKING([for libbfd])
   AC_ARG_WITH(bfd,
      AC_HELP_STRING(
         [--with-bfd@<:@=DIR@:>@],
         [specify where to find BFD libraries and includes]
      ),
      [bfd_paths="${withval}"],
      [bfd_paths="${bfd_default_paths}"]
   )
   for bfd_home_dir in [${bfd_paths} "not found"]; do
      if test -r "${bfd_home_dir}/lib${BITS}/libbfd.so" ; then
         BFD_LIBSDIR="${bfd_home_dir}/lib${BITS}"
         break
      elif test -r "${bfd_home_dir}/lib${BITS}/libbfd.a" ; then
         BFD_LIBSDIR="${bfd_home_dir}/lib${BITS}"
         break
      elif test -r "${bfd_home_dir}/lib/libbfd.so" ; then
         BFD_LIBSDIR="${bfd_home_dir}/lib"
         break
      elif test -r "${bfd_home_dir}/lib/libbfd.a" ; then
         BFD_LIBSDIR="${bfd_home_dir}/lib"
         break
      fi
   done
   AC_MSG_RESULT(${bfd_home_dir})

   AX_FLAGS_SAVE()
   CFLAGS="-I${bfd_home_dir}/include ${CFLAGS}"
   AC_CHECK_HEADERS([bfd.h], [BFD_INSTALLED="yes"], [BFD_INSTALLED="no"])
   AX_FLAGS_RESTORE()

   if test "${BFD_INSTALLED}" = "yes" ; then

      AC_MSG_CHECKING([for libiberty])
      AC_ARG_WITH(liberty,
         AC_HELP_STRING(
            [--with-liberty@<:@=DIR@:>@],
            [specify where to find LIBERTY libraries and includes]
         ),
         [liberty_paths="${withval}"],
         [liberty_paths="${libiberty_default_paths}"]
      )
      for liberty_home_dir in [${liberty_paths} "not found"]; do
         if test -r "${liberty_home_dir}/lib${BITS}/libiberty.so" ; then
            LIBERTY_LIBSDIR="${liberty_home_dir}/lib${BITS}"
            break
         elif test -r "${liberty_home_dir}/lib${BITS}/libiberty.a" ; then
            LIBERTY_LIBSDIR="${liberty_home_dir}/lib${BITS}"
            break
         elif test -r "${liberty_home_dir}/lib/libiberty.so" ; then
            LIBERTY_LIBSDIR="${liberty_home_dir}/lib"
            break
         elif test -r "${liberty_home_dir}/lib/libiberty.a" ; then
            LIBERTY_LIBSDIR="${liberty_home_dir}/lib"
            break
         fi
      done
      AC_MSG_RESULT(${liberty_home_dir})
      
      dnl Check if they work
      
      if test "${liberty_home_dir}" != "not found" ; then
         AC_MSG_CHECKING([if libbfd and libiberty works])

         AX_FLAGS_SAVE()
         if test "${OperatingSystem}" != "aix" ; then
            LIBS="-L${BFD_LIBSDIR} -lbfd -L${LIBERTY_LIBSDIR} -liberty ${LIBZ_LDFLAGS} ${LIBZ_LIBS}"
         else
            LIBS="-L${BFD_LIBSDIR} -lbfd -L${LIBERTY_LIBSDIR} -liberty ${LIBZ_LDFLAGS} ${LIBZ_LIBS} -lintl"
         fi
         CFLAGS="-I${bfd_home_dir}/include ${CFLAGS}"
         AC_TRY_LINK(
            [ #include <bfd.h> ], 
            [ bfd *abfd = bfd_openr ("", ""); ],
            [ bfd_liberty_works="yes" ]
         )

         if test "${bfd_liberty_works}" != "yes" ; then
            dnl On some machines BFD/LIBERTY need an special symbol (e.g BGL)
            AC_TRY_LINK(
               [ #include <bfd.h> 
                 int *__errno_location(void) { return 0; }
               ], 
               [ bfd *abfd = bfd_openr ("", ""); ],
               [ bfd_liberty_works="yes" ]
            )
            if test "${bfd_liberty_works}" = "yes" ; then
               AC_DEFINE([NEED_ERRNO_LOCATION_PATCH], 1, [Define to 1 if system requires __errno_location and does not provide it])
            fi
         fi

         if test "${bfd_liberty_works}" = "yes" ; then
            AC_MSG_RESULT([yes])

            BFD_HOME="${bfd_home_dir}"
            BFD_INCLUDES="${BFD_HOME}/include"
            BFD_CFLAGS="-I${BFD_INCLUDES}"
            BFD_CXXFLAGS=${BFD_CFLAGS}
            BFD_CPPFLAGS=${BFD_CFLAGS}
            BFD_LIBS="-lbfd"
            BFD_LDFLAGS="-L${BFD_LIBSDIR}"
            AC_SUBST(BFD_HOME)
            AC_SUBST(BFD_INCLUDES)
            AC_SUBST(BFD_CFLAGS)
            AC_SUBST(BFD_CXXFLAGS)
            AC_SUBST(BFD_CPPFLAGS)
            AC_SUBST(BFD_LIBS)
            AC_SUBST(BFD_LIBSDIR)
            if test -d ${BFD_LIBSDIR}/shared ; then
               BFD_SHAREDLIBSDIR="${BFD_LIBSDIR}/shared"
            else
               BFD_SHAREDLIBSDIR=${BFD_LIBSDIR}
            fi
            AC_SUBST(BFD_SHAREDLIBSDIR)
            AC_SUBST(BFD_LDFLAGS)

            LIBERTY_HOME="${liberty_home_dir}"
            LIBERTY_INCLUDES="${LIBERTY_HOME}/include"
            LIBERTY_CFLAGS="-I${LIBERTY_INCLUDES}"
            LIBERTY_CXXFLAGS=${LIBERTY_CFLAGS}
            LIBERTY_CPPFLAGS=${LIBERTY_CFLAGS}
            if test "${OperatingSystem}" != "aix" ; then
               LIBERTY_LIBS="-liberty"
            else
               LIBERTY_LIBS="-liberty -lintl"
            fi
            LIBERTY_LDFLAGS="-L${LIBERTY_LIBSDIR}"
            AC_SUBST(LIBERTY_HOME)
            AC_SUBST(LIBERTY_INCLUDES)
            AC_SUBST(LIBERTY_CFLAGS)
            AC_SUBST(LIBERTY_CXXFLAGS)
            AC_SUBST(LIBERTY_CPPFLAGS)
            AC_SUBST(LIBERTY_LIBS)
            AC_SUBST(LIBERTY_LIBSDIR)
            if test -d ${LIBERTY_LIBSDIR}/shared ; then
               BFD_SHAREDLIBSDIR="${LIBERTY_LIBSDIR}/shared"
            else
               BFD_SHAREDLIBSDIR=${LIBERTY_LIBSDIR}
            fi
            AC_SUBST(LIBERTY_SHAREDLIBSDIR)
            AC_SUBST(LIBERTY_LDFLAGS)

            BFD_INSTALLED="yes"
            LIBERTY_INSTALLED="yes"

            AC_DEFINE([HAVE_BFD], 1, [Define to 1 if BFD is installed in the system])

            AC_MSG_CHECKING([whether bfd_get_section_size is defined in bfd.h])
            AC_TRY_LINK(
              [ #include <bfd.h> ],
              [ 
                  asection *section;
                  int result = bfd_get_section_size(section); 
              ],
              [ bfd_get_section_size_found="yes"]
            )
            if test "${bfd_get_section_size_found}" = "yes" ; then
               AC_DEFINE(HAVE_BFD_GET_SECTION_SIZE, [], [Defined to 1 if bfd.h defines bfd_get_section_size])
               AC_MSG_RESULT([yes])
            else
               AC_MSG_RESULT([no])
            fi

            AC_MSG_CHECKING([whether bfd_get_section_size_before_reloc is defined in bfd.h])
            AC_TRY_LINK(
              [ #include <bfd.h> ],
              [ 
                  asection *section;
                  int result = bfd_get_section_size_before_reloc(section); 
              ],
              [ bfd_get_section_size_before_reloc_found="yes"]
            )
            if test "${bfd_get_section_size_before_reloc_found}" = "yes" ; then
               AC_DEFINE(HAVE_BFD_GET_SECTION_SIZE_BEFORE_RELOC, [], [Defined to 1 if bfd.h defines bfd_get_section_size_before_reloc])
               AC_MSG_RESULT([yes])
            else
               AC_MSG_RESULT([no])
            fi
         else
            AC_MSG_RESULT([no, see config.log for further details])
         fi
         AX_FLAGS_RESTORE()
      fi
   fi

   if test "${unwind_paths}" != "no"; then
      if test "${BFD_INSTALLED}" = "no" -o "${LIBERTY_INSTALLED}" = "no" ; then
         AC_MSG_ERROR([You have asked to gather call-site information through --with-unwind however either BFD or LIBERY are not found/installed. Please make sure that the binutils package is installed in its development form. The latest source can be downloaded from http://www.gnu.org/software/binutils])
      fi
   fi
])


# AX_CHECK_PERUSE
# ---------------------------
AC_DEFUN([AX_CHECK_PERUSE],
[
   AC_REQUIRE([AX_PROG_MPI])

   PERUSE_AVAILABILITY="no"
   AC_ARG_ENABLE(peruse,
      AC_HELP_STRING(
         [--enable-peruse],
         [Enable gathering information with PerUse]
      ),
      [enable_peruse="${enableval}"],
      [enable_peruse="auto"]
   )

   if test "${MPI_INSTALLED}" = "yes"; then
      if test "${enable_peruse}" = "auto" ; then
         AC_MSG_CHECKING(for peruse.h)
         if test -r ${MPI_INCLUDES}/peruse.h ; then
            AC_MSG_RESULT([available])
            enable_peruse="yes"
         else
            AC_MSG_RESULT([not available])
            enable_peruse="no"
         fi
      elif test "${enable_peruse}" = "yes" ; then
            AC_MSG_CHECKING(for peruse.h)
         if test -r ${MPI_INCLUDES}/peruse.h ; then
            AC_MSG_RESULT([available])
         else
            AC_MSG_NOTICE([Can not find the peruse header inside the MPI include directory.])
            AC_MSG_ERROR([Feature requested by the user but not available!])
         fi
      fi
   else
      enable_peruse="no"
   fi

   if test "${enable_peruse}" = "yes" ; then
      AC_MSG_CHECKING(for PERUSE_SUCCESS in peruse.h)
      AX_FLAGS_SAVE()
      CFLAGS="-I${MPI_INCLUDES}"
      AC_LANG_SAVE()
      AC_LANG([C])
      AC_TRY_COMPILE(
         [#include <peruse.h>],
         [
            int i = PERUSE_SUCCESS;
            return 0;
         ],
         [peruse_success="yes"],
         [peruse_success="no"]
      )
      AX_FLAGS_RESTORE()
      AC_LANG_RESTORE()

      if test "${peruse_success}" = "yes"; then
         AC_MSG_RESULT([available])
         AC_DEFINE([PERUSE_ENABLED], 1, [Determine if the PerUse API can be used])
         PERUSE_AVAILABILITY="yes"
      else
         AC_MSG_NOTICE([Can not find PERUSE_SUCCESS in peruse.h])
         AC_MSG_ERROR([Feature requested by the user but not available!])
      fi
   fi
])

# AX_PROG_MX
# ----------
AC_DEFUN([AX_PROG_MX],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(mx,
      AC_HELP_STRING(
         [--with-mx@<:@=DIR@:>@],
         [specify where to find MX libraries and includes]
      ),
      [mx_paths="$withval"],
      [mx_paths="/gpfs/apps/MX /opt/osshpc/mx"] dnl List of possible default paths
   )

   dnl Search for MX installation
   AX_FIND_INSTALLATION([MX], [$mx_paths], [mx])

   if test "$MX_INSTALLED" = "yes" ; then
      AC_CHECK_HEADERS([myriexpress.h], [], [MX_INSTALLED="no"])
      AC_CHECK_LIB([myriexpress], [mx_get_info], 
         [ 
           MX_LDFLAGS="${MX_LDFLAGS} -lmyriexpress"
           AC_SUBST(MX_LDFLAGS)
         ], 
         [ MX_INSTALLED="no" ]
      )
      AC_CHECK_HEADERS([mx_dispersion.h], [mx_dispersion_h_found="yes"], [mx_dispersion_h_found="no"])
      AC_CHECK_LIB([myriexpress], [mx_get_dispersion_counters], 
         [mx_get_dispersion_counters_found="yes"], 
         [mx_get_dispersion_counters="no"]
      )
      if test "$mx_dispersion_h_found" = "yes" -a "$mx_get_dispersion_counters_found" = "yes" ; then
         MX_CFLAGS="${MX_CFLAGS} -DMX_MARENOSTRUM_API"
         AC_SUBST(MX_CFLAGS)
         MX_CXXFLAGS="${MX_CFLAGS} -DMX_MARENOSTRUM_API"
         AC_SUBST(MX_CXXFLAGS)
      fi
   fi

   dnl Did the checks pass?
   AM_CONDITIONAL(HAVE_MX, test "${MX_INSTALLED}" = "yes")

   if test "$MX_INSTALLED" = "no" ; then
      AC_MSG_WARN([Myrinet MX counters tracing has been disabled])
   fi

   AX_FLAGS_RESTORE()
])

# AX_PROG_COUNTERS
# ----------------
AC_DEFUN([AX_PROG_COUNTERS],
[
   AC_REQUIRE([AX_PROG_PMAPI])
   AC_REQUIRE([AX_PROG_PAPI])

   if test "${papi_paths}" = "not_set" ; then
      if test "${target_os}" = "aix" ; then
         if test "${enable_pmapi}" = "not_set" ; then
            AC_MSG_ERROR([Attention! You're not indicating where to locate PAPI and if you want to use PMAPI. PAPI and PMAPI (specifically in AIX) allows gathering hardware performance counters. These counters are very useful to increase the richness of the final analysis. Please, use either --with-papi=DIR where DIR is the base location of the PAPI package or use --without-papi if you don't want to use PAPI in this installation. If you want to use PMAPI, please use --enable-pmapi, otherwise use --disable-pmapi.])
         fi
      else
         AC_MSG_ERROR([Attention! You're not indicating where to locate PAPI. PAPI allows gathering hardware performance counters. These counters are very useful to increase the richness of the final analysis. Please, use either --with-papi=DIR where DIR is the base location of the PAPI package or use --without-papi if you don't want to use PAPI in this installation.])
      fi
   fi

   if test "${PMAPI_ENABLED}" = "yes" -o "${PAPI_ENABLED}" = "yes" ; then
      AC_DEFINE([USE_HARDWARE_COUNTERS], 1, [Enable HWC support])
      use_hw_counters="1"
   else
      AC_DEFINE([USE_HARDWARE_COUNTERS], 0, [Disable HWC support])
      use_hw_counters="0"
   fi

   if test "${PMAPI_ENABLED}" = "yes" -a "${PAPI_ENABLED}" = "yes" ; then
      AC_MSG_ERROR([Error! Cannot use PMAPI and PAPI at the same time to access hardware counters!])
   fi
])


# AX_PROG_PMAPI
# -------------
AC_DEFUN([AX_PROG_PMAPI],
[
   AC_ARG_ENABLE(pmapi,
      AC_HELP_STRING(
         [--enable-pmapi],
         [Enable PMAPI library to gather CPU performance counters]
      ),
      [enable_pmapi="${enableval}"],
      [enable_pmapi="not_set"]
   )
   PMAPI_ENABLED="no"

   if test "${enable_pmapi}" = "yes" ; then
      PMAPI_ENABLED="yes"
      AC_CHECK_HEADERS([pmapi.h], [], [pmapi_h_notfound="yes"])

      if test "${pmapi_h_notfound}" = "yes" ; then
         AC_MSG_ERROR([Error! Unable to find pmapi.h])
      fi
   fi

   AM_CONDITIONAL(HAVE_PMAPI, test "${PMAPI_ENABLED}" = "yes")

   if test "${PMAPI_ENABLED}" = "yes" ; then
      AC_DEFINE([PMAPI_COUNTERS], [1], [PAPI is used as API to gain access to CPU hwc])
   else
      if test "${enable_pmapi}" = "yes" ; then
         AC_MSG_ERROR([Error PMAPI was not found and was enabled at configure time!])
      fi
   fi
])


# AX_PROG_PAPI
# ------------
AC_DEFUN([AX_PROG_PAPI],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(papi,
      AC_HELP_STRING(
         [--with-papi@<:@=DIR@:>@],
         [specify where to find PAPI libraries and includes]
      ),
      [papi_paths="${withval}"],
      [papi_paths="not_set"] dnl List of possible default paths
   )

   AC_ARG_ENABLE(sampling,
      AC_HELP_STRING(
         [--enable-sampling],
         [Enable PAPI sampling support]
      ),
      [enable_sampling="${enableval}"],
      [enable_sampling="auto"]
   )
   PAPI_SAMPLING_ENABLED="no"

   dnl Search for PAPI installation
   AX_FIND_INSTALLATION([PAPI], [$papi_paths], [papi])

   PAPI_ENABLED="${PAPI_INSTALLED}"
   if test "${PAPI_ENABLED}" = "yes" ; then
      AC_CHECK_HEADERS([papi.h], [], [papi_h_notfound="yes"])

      if test "${papi_h_notfound}" = "yes" ; then
         AC_MSG_ERROR([Error! Unable to find papi.h])
      fi

      if test "${IS_BGL_MACHINE}" = "yes" ; then
         LIBS="-static -lpapi -L${BG_HOME}/bglsys/lib -lbgl_perfctr.rts -ldevices.rts -lrts.rts"
      elif test "${IS_BGP_MACHINE}" = "yes" ; then
         LIBS="-static -lpapi -L${BG_HOME}/runtime/SPI -lSPI.cna"
      else
         if test "${OperatingSystem}" = "freebsd" ; then
            LIBS="-lpapi -lpmc"
         elif test "${OperatingSystem}" = "linux" -a "${Architecture}" = "powerpc" ; then
            LIBS="-lpapi -lperfctr"
            if test -d "${PAPI_HOME}/perfctr/lib" ; then
               LIBS="-L${PAPI_HOME}/perfctr/lib ${LIBS}"
            fi
         elif test "${OperatingSystem}" = "aix" -a "${Architecture}" = "powerpc" ; then
            if test "${BITS}" = "64" ; then
               if test -f "${PAPI_LIBSDIR}/libpapi64.a" -o -f "${PAPI_LIBSDIR}/libpapi64.so" ; then
                  LIBS="-lpapi64 -lpmapi"
               else
                  LIBS="-lpapi -lpmapi"
               fi 
            else
               LIBS="-lpapi -lpmapi"
            fi
         else
            LIBS="-lpapi"
         fi
      fi

      AC_CHECK_LIB([papi], [PAPI_start],
         [ 
            PAPI_LIBS="${LIBS}"
            AC_SUBST(PAPI_LIBS)
         ],
         [PAPI_ENABLED="no"]
      )
   fi

   AM_CONDITIONAL(HAVE_PAPI, test "${PAPI_ENABLED}" = "yes")

   AC_DEFINE([SAMPLING_SUPPORT], [1], [Enable Sampling])

   if test "${PAPI_ENABLED}" = "yes" ; then
      PAPI_SAMPLING_ENABLED="yes"
      AC_DEFINE([PAPI_COUNTERS], [1], [PAPI is used as API to gain access to CPU hwc])
      AC_DEFINE([NEW_HWC_SYSTEM], [1], [Enable HWC support])
      AC_MSG_NOTICE([PAPI and substrate libraries: ${PAPI_LIBS}])
      if test "${enable_sampling}" = "yes" ; then
         AC_CHECK_MEMBER([PAPI_substrate_info_t.supports_hw_overflow],[support_hw_overflow="yes"],[support_hw_overflow="no"],[#include <papi.h>])
         if test "${support_hw_overflow}" = "yes" ; then
            AC_DEFINE([HAVE_SUPPORT_HW_OVERFLOW], [1], [Use supports_hw_overflow field])
            AC_DEFINE([PAPI_SAMPLING_SUPPORT], [1], [Enable PAPI sampling support])
         else
            AC_CHECK_MEMBER([PAPI_substrate_info_t.hardware_intr_sig],[hardware_intr_sig="yes"],[hardware_intr_sig="no"],[#include <papi.h>])
            if test "${hardware_intr_sig}" = "yes" ; then
               AC_DEFINE([HAVE_HARDWARE_INTR_SIG], [1], [Use hardware_intr_sig field])
               AC_DEFINE([PAPI_SAMPLING_SUPPORT], [1], [Enable PAPI sampling support])
            else
               AC_CHECK_MEMBER([PAPI_component_info_t.hardware_intr],[support_comp_hw="yes"],[support_comp_hw="no"],[#include <papi.h>])
               if test "${support_comp_hw}" = "yes" ; then
                  AC_DEFINE([HAVE_COMPONENT_HARDWARE_INTR], [1], [Use hardware_intr in PAPI_component_info_t field])
                  AC_DEFINE([PAPI_SAMPLING_SUPPORT], [1], [Enable PAPI sampling support])
               else
                  AC_MSG_NOTICE([Cannot determine how to check whether PAPI supports HW overflows! Will use software-based sampling.])
               fi
            fi
         fi
      fi
   else
      if test "${papi_paths}" != "no" -a "${papi_paths}" != "not_set"; then
         AC_MSG_ERROR([Error PAPI was not found and was enabled at configure time!])
      fi
   fi

   AX_FLAGS_RESTORE()
])

# AX_IS_ALTIX_MACHINE
# ----------------
AC_DEFUN([AX_IS_ALTIX_MACHINE],
[
   AC_MSG_CHECKING([if this is an Altix machine])
   if test -r /etc/sgi-release ; then 
      AC_MSG_RESULT([yes])
      IS_ALTIX_MACHINE="yes"
			AC_DEFINE([IS_ALTIX], 1, [Defined if this machine is a SGI Altix])
   else
      AC_MSG_RESULT([no])
      IS_ALTIX_MACHINE="no"
   fi
])

# AX_HAVE_MMTIMER_DEVICE
# ----------------
AC_DEFUN([AX_HAVE_MMTIMER_DEVICE],
[
   AC_REQUIRE([AX_IS_ALTIX_MACHINE])

   if test "${IS_ALTIX_MACHINE}" = "yes" ; then
      AC_MSG_CHECKING([if this is an Altix machine has MMTimer device])
      if test -r /dev/mmtimer ; then 
         AC_MSG_RESULT([yes])
         AC_DEFINE([HAVE_MMTIMER_DEVICE], 1, [Defined if this machine has a MMTimer device and it is readable])
         HAVE_MMTIMER_DEVICE="yes"
      else
         AC_MSG_RESULT([no])
         HAVE_MMTIMER_DEVICE="no"
      fi
   else
      HAVE_MMTIMER_DEVICE="no"
   fi
])

# AX_IS_CELL_MACHINE
# --------------------
AC_DEFUN([AX_IS_CELL_MACHINE],
[
   AC_MSG_CHECKING([if this is a CELL machine])
   AC_ARG_ENABLE(check-cell,
      AC_HELP_STRING(
         [--enable-check-cell],
         [Enable check to known if this is a Cell BE machine (enabled by default)]
      ),
      [enable_check_cell="${enableval}"],
      [enable_check_cell="yes"]
   )
   grep "Cell Broadband Engine" /proc/cpuinfo 2> /dev/null > /dev/null
   IS_CELL_MACHINE=$?
   if test "${enable_check_cell}" = "yes" -a "${IS_CELL_MACHINE}" = "0" ; then
      AC_MSG_RESULT([yes])
      AC_DEFINE([IS_CELL_MACHINE], 1, [Defined if this machine is a CELL machine])
      if test -f /usr/include/libspe2.h ; then
         AC_DEFINE([CELL_SDK], 2, [Defined if this machine has SDK x installed])
         CELL_SDK="2"
      elif test -f /usr/include/libspe.h ; then
         AC_DEFINE([CELL_SDK], 1, [Defined if this machine has SDK x installed])
         CELL_SDK="1"
      else
         AC_MSG_ERROR([Cannot determine which CELL SDK is installed])
         CELL_SDK="0"
      fi
      IS_CELL_MACHINE="yes"
   else
      AC_MSG_RESULT([no])
      IS_CELL_MACHINE="no"
   fi
   AM_CONDITIONAL(IS_CELL_MACHINE, test "${IS_CELL_MACHINE}" = "yes")
   AM_CONDITIONAL(CELL_SDK_2, test "${CELL_SDK}" = "2")
   AM_CONDITIONAL(CELL_SDK_1, test "${CELL_SDK}" = "1")

   if test "${CELL_SDK}" = "1" ; then
      AC_MSG_NOTICE([CBEA SDK installed support 1.x])
   elif test "${CELL_SDK}" = "2" ; then
      AC_MSG_NOTICE([CBEA SDK installed support 2.x/3.x])
   fi
])

# AX_IS_CRAY_XT
# ---------------------
AC_DEFUN([AX_IS_CRAY_XT],
[
   AC_MSG_CHECKING([if this is a Cray XT machine])
   AC_ARG_ENABLE(check-cray-xt,
      AC_HELP_STRING(
         [--enable-check-cray-xt],
         [Enable check to known if this is a frontend to a Cray XT machine (enabled by default)]
      ),
      [enable_check_cxt="${enableval}"],
      [enable_check_cxt="yes"]
   )

   IS_CXT_MACHINE="no"
   if test "${enable_check_cxt}" = "yes" ; then
      if test -d /opt/cray ; then
         if test `which cc | grep xt-asyncpe | wc -l` != "0" ; then
           IS_CXT_MACHINE="yes"
         fi
      fi
   fi
   AC_MSG_RESULT([$IS_CXT_MACHINE])
   AM_CONDITIONAL(IS_CRAY_XT_MACHINE, test "${IS_CXT_MACHINE}" = "yes")
])

# AX_IS_BGP_MACHINE
# ---------------------
AC_DEFUN([AX_IS_BGP_MACHINE],
[
   AC_MSG_CHECKING([if this is a BG/P machine])
   AC_ARG_ENABLE(check-bgp,
      AC_HELP_STRING(
         [--enable-check-bgp],
         [Enable check to known if this is a frontend to a BG/P BE machine (enabled by default)]
      ),
      [enable_check_bgp="${enableval}"],
      [enable_check_bgp="yes"]
   )

   if test "${enable_check_bgp}" = "yes" -a -d /bgsys/drivers/ppcfloor ; then
     IS_BGP_MACHINE="yes"
     BG_HOME="/bgsys/drivers/ppcfloor"
     CFLAGS="${CFLAGS} -I${BG_HOME}/bglsys/include -I${BG_HOME}/arch/include -I${BG_HOME}/blrts-gnu/include"
     AC_SUBST(BG_HOME)
     AC_MSG_RESULT([yes])
     AC_DEFINE([IS_BGP_MACHINE], 1, [Defined if this machine is a BG/P machine])
   else
     IS_BGP_MACHINE="no"
     AC_MSG_RESULT([no])
   fi
   AM_CONDITIONAL(IS_BGP_MACHINE, test "${IS_BGP_MACHINE}" = "yes")
])

# AX_IS_BGL_MACHINE
# ---------------------
AC_DEFUN([AX_IS_BGL_MACHINE],
[
   AC_MSG_CHECKING([if this is a BG/L machine])
   AC_ARG_ENABLE(check-bgl,
      AC_HELP_STRING(
         [--enable-check-bgl],
         [Enable check to known if this is a frontend to a BG/L BE machine (enabled by default)]
      ),
      [enable_check_bgl="${enableval}"],
      [enable_check_bgl="yes"]
   )

   if test "${enable_check_bgl}" = "yes" -a -d /bgl/BlueLight/ppcfloor/bglsys ; then
     IS_BGL_MACHINE="yes"
     BG_HOME="/bgl/BlueLight/ppcfloor"
     CFLAGS="${CFLAGS} -I${BG_HOME}/bglsys/include -I${BG_HOME}/blrts-gnu/include"
     AC_SUBST(BG_HOME)
     AC_MSG_RESULT([yes])
     AC_DEFINE([IS_BGL_MACHINE], 1, [Defined if this machine is a BG/L machine])
   else
     IS_BGL_MACHINE="no"
     AC_MSG_RESULT([no])
   fi
   AM_CONDITIONAL(IS_BGL_MACHINE, test "${IS_BGL_MACHINE}" = "yes")
])

# AX_IS_MN_MACHINE
#---------------------
AC_DEFUN([AX_IS_MN_MACHINE],
[
   AC_ARG_ENABLE(check-mn,
      AC_HELP_STRING(
         [--enable-check-mn],
         [Enable check to known if this is MareNostrum (enabled by default)]
      ),
      [enable_check_mn="${enableval}"],
      [enable_check_mn="yes"]
   )
   AC_MSG_CHECKING([if this is MN machine])
   grep "Welcome to MareNostrum" /etc/motd 2> /dev/null > /dev/null
   GREP_RESULT=$?
   if test "${enable_check_mn}" = "yes" -a "${GREP_RESULT}" = "0" ; then
      AC_MSG_RESULT([yes])
      AC_DEFINE([IS_MN_MACHINE], 1, [Defined if this machine is MN])
      IS_MN_MACHINE="yes"
   else
      AC_MSG_RESULT([no])
      IS_MN_MACHINE="no"
   fi
])

# AX_OPENMP
#-----------------
AC_DEFUN([AX_OPENMP],
[
   AC_PREREQ(2.59)

   AC_CACHE_CHECK([for OpenMP flag of _AC_LANG compiler],
      ax_cv_[]_AC_LANG_ABBREV[]_openmp,
      [save[]_AC_LANG_PREFIX[]FLAGS=$[]_AC_LANG_PREFIX[]FLAGS ax_cv_[]_AC_LANG_ABBREV[]_openmp=unknown
      # Flags to try:  -fopenmp (gcc), -openmp (icc), -mp (SGI &amp; PGI),
      #                -xopenmp (Sun), -omp (Tru64), -qsmp=omp (AIX), none
      ax_openmp_flags="-fopenmp -openmp -mp -xopenmp -omp -qsmp=omp none"
      if test "x$OPENMP_[]_AC_LANG_PREFIX[]FLAGS" != x; then
         ax_openmp_flags="$OPENMP_[]_AC_LANG_PREFIX[]FLAGS $ax_openmp_flags"
      fi
      for ax_openmp_flag in $ax_openmp_flags; do
         case $ax_openmp_flag in
            none) []_AC_LANG_PREFIX[]FLAGS=$save[]_AC_LANG_PREFIX[] ;;
            *) []_AC_LANG_PREFIX[]FLAGS="$save[]_AC_LANG_PREFIX[]FLAGS $ax_openmp_flag" ;;
         esac
         AC_TRY_LINK_FUNC(omp_set_num_threads,
   	       [ax_cv_[]_AC_LANG_ABBREV[]_openmp=$ax_openmp_flag; break])
      done
      []_AC_LANG_PREFIX[]FLAGS=$save[]_AC_LANG_PREFIX[]FLAGS])
      if test "x$ax_cv_[]_AC_LANG_ABBREV[]_openmp" = "xunknown"; then
         m4_default([$2],:)
      else
         if test "x$ax_cv_[]_AC_LANG_ABBREV[]_openmp" != "xnone"; then
            OPENMP_[]_AC_LANG_PREFIX[]FLAGS=$ax_cv_[]_AC_LANG_ABBREV[]_openmp
         fi
         m4_default([$1], [AC_DEFINE(HAVE_OPENMP,1,[Define if OpenMP is enabled])])
      fi
])

# AX_CHECK_UNWIND
# ------------
AC_DEFUN([AX_CHECK_UNWIND],
[
   AX_FLAGS_SAVE()

   libunwind_works="no"

   AC_ARG_WITH(unwind,
      AC_HELP_STRING(
         [--with-unwind@<:@=DIR@:>@],
         [specify where to find Unwind libraries and includes]
      ),
      [unwind_paths=${withval}],
      [unwind_paths="not_set"]
   )

   if test "${unwind_paths}" = "not_set" ; then
      AC_MSG_ERROR([You haven't specified the location of the libunwind through the --with-unwind parameter. The libunwind library allows Extrae gathering information of call-site locations at OpenMP and MPI calls, at sample points or whenever the user wants to collect them through the Extrae API. This data is very useful to attribute to a certain region of code any performance issue. You can download libunwind from: https://savannah.nongnu.org/projects/libunwind - download version 1.0.1 or higher. If you don't want to use libunwind, you can pass --without-unwind.])
   fi

   if test "${unwind_paths}" != "no" ; then

      AX_FIND_INSTALLATION([UNWIND], [$unwind_paths], [unwind])

      if test "${UNWIND_INSTALLED}" = "yes" ; then 

         UNWIND_LIBS="-lunwind"
         AC_SUBST(UNWIND_LIBS)

         CFLAGS="${CFLAGS} ${UNWIND_CFLAGS}"
         LIBS="${LIBS} -lunwind"
         LDFLAGS="${LDFLAGS} ${UNWIND_LDFLAGS}"

         AC_MSG_CHECKING([if libunwind works])

         AC_TRY_LINK(
            [ #define UNW_LOCAL_ONLY
              #include <libunwind.h> ], 
            [ unw_cursor_t cursor;
              unw_context_t uc;
              unw_word_t ip;

              unw_getcontext(&uc);
              unw_init_local(&cursor, &uc);
              unw_step(&cursor);
              unw_get_reg(&cursor, UNW_REG_IP, &ip);
            ],
            [ libunwind_works="yes" ],
            [ libunwind_works="no" ]
         )

         AC_MSG_RESULT([${libunwind_works}])

      fi

      if test "${libunwind_works}" = "yes"; then
         AC_DEFINE([UNWIND_SUPPORT], [1], [Unwinding support enabled for IA64/x86-64])
         AC_DEFINE([HAVE_LIBUNWIND_H], [1], [Define to 1 if you have <libunwind.h> header file])
      else
         AC_MSG_ERROR([Cannot link libunwind example. Check that --with-unwind points to the appropriate libunwind directory.])
      fi
   fi
   AX_FLAGS_RESTORE()
])

# AX_CHECK_LIBZ
# ------------
AC_DEFUN([AX_CHECK_LIBZ],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(libz,
      AC_HELP_STRING(
         [--with-libz@<:@=DIR@:>@],
         [specify where to find libz libraries and includes]
      ),
      [libz_paths="${withval}"],
      [libz_paths="/usr/local /usr"] dnl List of possible default paths
   )

   for zhome_dir in [${libz_paths} "not found"]; do
      if test -f "${zhome_dir}/${BITS}/include/zlib.h" ; then 
         if test -f "${zhome_dir}/${BITS}/lib/libz.a" -o \
                 -f "${zhome_dir}/${BITS}/lib/libz.so" ; then
            LIBZ_HOME="${zhome_dir}/${BITS}"
            LIBZ_LIBSDIR="${zhome_dir}/${BITS}/lib"
            break
         fi
      elif test -f "${zhome_dir}/include/zlib.h" ; then
         if test -f "${zhome_dir}/lib${BITS}/libz.a" -o \
                 -f "${zhome_dir}/lib${BITS}/libz.so" ; then
            LIBZ_HOME="${zhome_dir}"
            LIBZ_LIBSDIR="${zhome_dir}/lib${BITS}"
            break
         fi
         if test -f "${zhome_dir}/lib/libz.a" -o \
                 -f "${zhome_dir}/lib/libz.so" ; then
            LIBZ_HOME="${zhome_dir}"
            LIBZ_LIBSDIR="${zhome_dir}/lib"
            break
         fi
      fi
    done

   LIBZ_INCLUDES="${LIBZ_HOME}/include"
   LIBZ_CFLAGS="-I${LIBZ_INCLUDES}"
   LIBZ_CPPFLAGS=${LIBZ_CFLAGS}
   LIBZ_CXXFLAGS=${LIBZ_CFLAGS}
   LIBZ_LIBS="-lz"
   LIBZ_LDFLAGS="-L${LIBZ_LIBSDIR}"
   if test -d ${LIBZ_LIBSDIR}/shared ; then 
      LIBZ_SHAREDLIBSDIR="${LIBZ_LIBSDIR}/shared"
   else
      LIBZ_SHAREDLIBSDIR=${LIBZ_LIBSDIR}
   fi

   AC_SUBST(LIBZ_HOME)
   AC_SUBST(LIBZ_CFLAGS)
   AC_SUBST(LIBZ_CPPFLAGS)
   AC_SUBST(LIBZ_CXXFLAGS)
   AC_SUBST(LIBZ_INCLUDES)
   AC_SUBST(LIBZ_LIBSDIR)
   AC_SUBST(LIBZ_SHAREDLIBSDIR)
   AC_SUBST(LIBZ_LIBS)
   AC_SUBST(LIBZ_LDFLAGS)

   CFLAGS="${CFLAGS} ${LIBZ_CFLAGS}"
   LIBS="${LIBS} ${LIBZ_LIBS}"
   LDFLAGS="${LDFLAGS} ${LIBZ_LDFLAGS}"

   AC_CHECK_LIB(z, inflateEnd, [zlib_cv_libz=yes], [zlib_cv_libz=no])
   AC_CHECK_HEADER(zlib.h, [zlib_cv_zlib_h=yes], [zlib_cv_zlib_h=no])

   if test "${zlib_cv_libz}" = "yes" -a "${zlib_cv_zlib_h}" = "yes" ; then
      AC_DEFINE([HAVE_ZLIB], [1], [Zlib available])
			ZLIB_INSTALLED="yes"
   else
      ZLIB_INSTALLED="no"
   fi

   AM_CONDITIONAL(HAVE_ZLIB, test "${ZLIB_INSTALLED}" = "yes")

   AX_FLAGS_RESTORE()
])

# AX_PROG_LIBEXECINFO
# -------------
AC_DEFUN([AX_PROG_LIBEXECINFO],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(libexecinfo,
      AC_HELP_STRING(
         [--with-libexecinfo=@<:@=DIR@:>@],
         [specify where to find libexecinfo libraries and includes (FreeBSD specific?)]
      ),
      [libexecinfo_paths="${withval}"],
      [libexecinfo_paths="no"]
   )

   if test "${libexecinfo_paths}" != "no" ; then
      AX_FIND_INSTALLATION([libexecinfo], [${libexecinfo_paths}], [libexecinfo])
      if test "${LIBEXECINFO_INSTALLED}" = "yes" ; then
        if test -f ${LIBEXECINFO_HOME}/lib/libexecinfo.a -o \
                -f ${LIBEXECINFO_HOME}/lib/libexecinfo.so ; then
           if test -f ${LIBEXECINFO_HOME}/include/execinfo.h ; then
              CFLAGS="-I ${LIBEXECINFO_HOME}/include"
              LIBS="-L ${LIBEXECINFO_HOME}/lib -lexecinfo"
		          AC_TRY_LINK(
		              [ #include <execinfo.h> ],
		              [ backtrace ((void*)0, 0); ],
		              [ libexecinfo_links="yes" ]
		            )
              if test "${libexecinfo_links}" = "yes" ; then
                 AC_DEFINE([HAVE_EXECINFO_H], 1, [Define to 1 if you have the <execinfo.h> header file.])
              else
                 AC_MSG_ERROR([Cannot link using libexecinfo... See config.log for further details])
              fi
           else
              AC_MSG_ERROR([Cannot find libexecinfo header files in ${libexecinfo_paths}/include, maybe you can install it from /usr/ports/devel/libexecinfo])
           fi
        else
           AC_MSG_ERROR([Cannot find libexecinfo library files in ${libexecinfo_paths}/lib, maybe you can install it from /usr/ports/devel/libexecinfo])
        fi
      fi
   fi

   AX_FLAGS_RESTORE()
])

# AX_PROG_LIBDWARF
# -------------
AC_DEFUN([AX_PROG_LIBDWARF],
[
   libdwarf_found="no"
   AX_FLAGS_SAVE()

   AC_ARG_WITH(dwarf,
      AC_HELP_STRING(
         [--with-dwarf=@<:@=DIR@:>@],
         [specify where to find dwarf libraries and includes]
      ),
      [dwarf_paths="${withval}"],
      [dwarf_paths="no"]
   )

   if test "${dwarf_paths}" != "no" ; then
      AX_FIND_INSTALLATION([DWARF], [${dwarf_paths}], [dwarf])
      if test "${DWARF_INSTALLED}" = "yes" ; then
        if test -f ${DWARF_LIBSDIR}/libdwarf.a -o \
                -f ${DWARF_LIBSDIR}/libdwarf.so ; then
           if test -f ${DWARF_HOME}/include/libdwarf.h -a \
                   -f ${DWARF_HOME}/include/dwarf.h ; then
              libdwarf_found="yes"
           elif test -f ${DWARF_HOME}/include/libdwarf/libdwarf.h -a \
                     -f ${DWARF_HOME}/include/libdwarf/dwarf.h ; then
              libdwarf_found="yes"
           else
              AC_MSG_ERROR([Cannot find DWARF header files in ${dwarf_paths}/include])
           fi
        else
           AC_MSG_ERROR([Cannot find DWARF library files in ${dwarf_paths}/lib])
        fi
      fi
   fi

   AX_FLAGS_RESTORE()
])

# AX_PROG_DYNINST
# -------------
AC_DEFUN([AX_PROG_DYNINST],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(dyninst,
      AC_HELP_STRING(
         [--with-dyninst@<:@=DIR@:>@],
         [specify where to find DynInst libraries and includes]
      ),
      [dyninst_paths="${withval}"],
      [dyninst_paths="not_set"]
   )

   if test "${dyninst_paths}" = "not_set" ; then
      AC_MSG_ERROR([Attention! You haven't specified the location for DynInst, DynInst is a library for instrumenting binaries and allows Extrae to modify the application to analyze without having to modify the application sources. To use DynInst you have to pass --with-dyninst with the location of the DynInst installation and also --with-dwarf with the location of the libdwarf package. You can download Dyninst from http://www.dyninst.org. If you are not interested on DynInst, simply pass --without-dyninst to the configure parameters.])
   fi

   if test "${libdwarf_found}" != "yes" -a "${dyninst_paths}" != "no" ; then
      AC_MSG_ERROR([Cannot add DynInst support without libdwarf. Check for --with-dwarf option])
   fi

   dnl Search for Dyninst installation
   AX_FIND_INSTALLATION([DYNINST], [${dyninst_paths}], [dyninst])

   if test "${DYNINST_INSTALLED}" = "yes" ; then
      AC_LANG_SAVE()

      AC_LANG_PUSH([C++])

      dnl Check for Dyninst header files.
      CXXFLAGS="${CXXFLAGS} -I${DYNINST_INCLUDES}"
      CPPFLAGS="${CPPFLAGS} -I${DYNINST_INCLUDES}"
      AC_CHECK_HEADERS([BPatch.h], [], [DYNINST_INSTALLED="no"])

      AC_LANG_RESTORE()
   fi

   dnl Did the checks pass?
   AM_CONDITIONAL(HAVE_DYNINST, test "${DYNINST_INSTALLED}" = "yes")

   if test "${DYNINST_INSTALLED}" = "no" -a "${dyninst_paths}" != "no"; then
      AC_MSG_ERROR([Dyninst cannot be found])
   else
      AC_DEFINE([HAVE_DYNINST], 1, [Define to 1 if DYNINST is installed in the system])
   fi

   AX_FLAGS_RESTORE()
])

# AX_PROG_MRNET
# -------------
AC_DEFUN([AX_PROG_MRNET],
[
	AX_FLAGS_SAVE()
	AC_LANG_SAVE()
	AC_LANG([C++])

	if test -f "src/tracer/mrnet/mrnet_root.C" ; then

		AC_ARG_WITH(mrnet,
			AC_HELP_STRING(
				[--with-mrnet@<:@=DIR@:>@],
				[specify where to find MRNet libraries and includes]
			),
			[mrnet_paths="$withval"],
			[mrnet_paths="/home/bsc41/bsc41127/mrnet_last"] dnl List of possible default paths
		)

		dnl Search for MRNet installation
		AX_FIND_INSTALLATION([MRNET], [$mrnet_paths], [mrnet])

		if test "$MRNET_INSTALLED" = "yes" ; then

			dnl Begin hack
			MRNET_CXXFLAGS="${MRNET_CXXFLAGS} -I${MRNET_HOME}/src/src -Dos_linux -I/gpfs/apps/BOOST/boost_1_33_1-64/include/boost-1_33_1"
			AC_SUBST(MRNET_CXXFLAGS)
			dnl End hack

			dnl Check for MRNet header files.
			CXXFLAGS="${CXXFLAGS} -I${MRNET_INCLUDES}/mrnet ${MRNET_CXXFLAGS}"
			CPPFLAGS="${CPPFLAGS} -I${MRNET_INCLUDES}/mrnet ${MRNET_CXXFLAGS}"
			AC_CHECK_HEADERS([MRNet.h], [], [MRNET_INSTALLED="no"])

			dnl Check for libraries.
			AC_MSG_CHECKING([for libmrnet and libxplat])

			if test -f ${MRNET_LIBSDIR}/libmrnet.a -a -f ${MRNET_LIBSDIR}/libxplat.a ; then
				MRNET_LIBS="-lmrnet -lxplat -lpthread -ldl"
				AC_SUBST(MRNET_LIBS)
				AC_MSG_RESULT([yes])
			else
				MRNET_INSTALLED="no"
				AC_MSG_RESULT([no])
			fi
		fi

		if test "${MRNET_INSTALLED}" = "no" ; then
			AC_MSG_WARN([MRNet support has been disabled])
		else
			AC_DEFINE([HAVE_MRNET], 1, [Define to 1 if MRNET is installed in the system])
	
			AX_FLAGS_RESTORE()
			dnl Check for clustering and spectral support
			AC_ARG_WITH(clustering,
				AC_HELP_STRING(
					[--with-clustering@<:@=DIR@:>@],
					[specify where to find clustering libraries and includes]
				),
				[clustering_paths="$withval"],
				[clustering_paths="/gpfs/apps/CEPBATOOLS/burst-clusterizer-devel"] dnl List of possible default paths
			)
			dnl Search for Clustering installation
			AX_FIND_INSTALLATION([CLUSTERING], [$clustering_paths], [clustering])

			if test "${CLUSTERING_INSTALLED}" = "yes" ; then
		        CLUSTERING_LIBS="-lMRNetClustering"

		        dnl FIXME: Can't do these checks because MRNetClustering includes types.h, which includes config.h, which is not distributed!
		        dnl AC_MSG_CHECKING([whether a program can be linked with libMRNetClustering])
		        dnl AC_TRY_LINK(
		        dnl [ /* #include <MRNetClustering.h> */ ],
		        dnl [ MRNetClustering *c = new MRNetClustering(); ],
		        dnl [ clustering_links="yes" ]
		        dnl )

		        AC_SUBST(CLUSTERING_LIBS)
				AC_DEFINE([HAVE_CLUSTERING], 1, [Define to 1 if CLUSTERING is installed in the system])
		    fi
			AX_FLAGS_RESTORE()

			AC_ARG_WITH(clustering-ann,
				AC_HELP_STRING(
					[--with-clustering-ann@<:@=DIR@:>@],
					[specify where to find ANN libraries and includes]
				),
				[clustering_ann_paths="$withval"],
				[clustering_ann_paths="/gpfs/apps/CEPBATOOLS/libMRNetANN"] dnl List of possible default paths
			)
			dnl Search for libANN installation
			AX_FIND_INSTALLATION([ANN], [$clustering_ann_paths], [clustering-ann])
			if test "${ANN_INSTALLED}" = "yes" ; then
				ANN_LIBS="-lMRNetANN"
				AC_SUBST(ANN_LIBS)
				AC_DEFINE([HAVE_ANN], 1, [Define to 1 if libANN is installed in the system])
			fi
			AX_FLAGS_RESTORE()

			AC_ARG_WITH(spectral,
				AC_HELP_STRING(
					[--with-spectral@<:@=DIR@:>@],
					[specify where to find spectral analysis libraries and includes]
				),
				[spectral_paths="$withval"],
				[spectral_paths="/home/bsc41/bsc41127/apps/spectral_last"] dnl List of possible default paths
			)
			dnl Search for Spectral Analysis installation
			AX_FIND_INSTALLATION([SPECTRAL], [$spectral_paths], [spectral])

			if test "${SPECTRAL_HOME}" != "" ; then

		        LIBS="-L${SPECTRAL_HOME} -lspectral" dnl No need once the installation follows the standard (-L)
		        CXXFLAGS="${CXXFLAGS} -I${SPECTRAL_HOME}"        dnl No need once the installation follows the standard

		        AC_MSG_CHECKING([whether libspectral has unresolved dependencies with libfft])
		        AC_TRY_LINK(
		            [ #include <stdio.h>
		              #include <signal_interface.h> ],
		            [ destroySignal(NULL); ],
		            [ spectral_links="yes" ]
		        )

		        if test "${spectral_links}" = "yes" ; then
		            AC_MSG_RESULT([no])
		        else
		            dnl There are unresolved dependencies with fftw3
		            AC_MSG_RESULT([yes])
		            AC_ARG_WITH(spectral-fft,
		                AC_HELP_STRING(
		                    [--with-spectral-fft@<:@=DIR@:>@],
		                    [specify where to find FFT libraries and includes]
		                ),
		                [fft_paths="$withval"],
		                [fft_paths="/gpfs/apps/FFTW/3.1.1"] dnl List of possible default paths
		            )
		            dnl Search for FFT installation
		            AX_FIND_INSTALLATION([FFT], [$fft_paths], [spectral-fft])
	
		            LIBS="${LIBS} ${FFT_LDFLAGS} -lfftw3 -lm"
	
		            AC_TRY_LINK(
		                [ #include <stdio.h>
		                  #include <signal_interface.h> ],
		                [ destroySignal(NULL); ],
		                [ spectral_links="yes" ]
		            )
		        fi
	
		        AC_MSG_CHECKING([whether a program can be linked with libspectral])
				if test "${spectral_links}" = "yes" ; then
					SPECTRAL_LIBS="${LIBS}"
					SPECTRAL_CXXFLAGS="-I${SPECTRAL_HOME}" dnl No need once the installation follows the standard
					AC_SUBST(SPECTRAL_LIBS)
					AC_SUBST(SPECTRAL_CXXFLAGS)   dnl No need once the installation follows the standard
					AC_MSG_RESULT([yes])
					AC_DEFINE([HAVE_SPECTRAL], 1, [Define to 1 if SPECTRAL ANALYSIS is installed in the system])
				else
					AC_MSG_RESULT([no])
				fi
			fi
		fi
	fi
	dnl Did the checks pass?
	AM_CONDITIONAL(HAVE_MRNET, test "${MRNET_INSTALLED}" = "yes")
	AM_CONDITIONAL(HAVE_CLUSTERING, test "${CLUSTERING_INSTALLED}" = "yes")
	AM_CONDITIONAL(HAVE_SPECTRAL, test "${spectral_links}" = "yes")
	
dnl	MRNET_SRCDIR=""
dnl	if test "${MRNET_INSTALLED}" = "yes" ; then
dnl		MRNET_SRCDIR="mrnet"
dnl	fi
dnl	AC_SUBST([MRNET_SRCDIR])

	AX_FLAGS_RESTORE()
	AC_LANG_RESTORE()
])

AC_DEFUN([AX_CHECK_WEAK_ALIAS_ATTRIBUTE],
[
  # Test whether compiler accepts __attribute__ form of weak aliasing
  AC_CACHE_CHECK([whether ${CC} accepts function __attribute__((weak,alias()))],
  [ax_cv_weak_alias_attribute], [

    # We add -Werror if it's gcc to force an error exit if the weak attribute
    # isn't understood

    save_CFLAGS=${CFLAGS}
    
    if test "${GCC}" = "yes" ; then
       CFLAGS="-Werror"
    elif test "`basename ${CC}`" = "xlc" ; then
       CFLAGS="-qhalt=i"
    fi

    if test "${pgi_compiler}" = "no" ; then
       # Try linking with a weak alias...
       AC_LINK_IFELSE([
         AC_LANG_PROGRAM([
            void __weakf(int c) {}
            void weakf(int c) __attribute__((weak, alias("__weakf")));],
            [weakf(0)])],
         [ax_cv_weak_alias_attribute="yes"],
         [ax_cv_weak_alias_attribute="no"])
     else
        ax_cv_weak_alias_attribute="no"
     fi

     # Restore original CFLAGS
     CFLAGS=${save_CFLAGS}
  ])

  # What was the result of the test?
  AS_IF([test "${ax_cv_weak_alias_attribute}" = "yes"],
  [
    AC_DEFINE([HAVE_WEAK_ALIAS_ATTRIBUTE], 1,
              [Define this if weak aliases may be created with __attribute__])
  ])
])

AC_DEFUN([AX_CHECK_ALIAS_ATTRIBUTE],
[
  # Test whether compiler accepts __attribute__ form of aliasing
  AC_CACHE_CHECK([whether ${CC} accepts function __attribute__((alias()))],
  [ax_cv_alias_attribute], [

    # We add -Werror if it's gcc to force an error exit if the weak attribute
    # isn't understood

    save_CFLAGS=${CFLAGS}
    
    if test "${GCC}" = "yes" ; then
       CFLAGS="-Werror"
    elif test "`basename ${CC}`" = "xlc" ; then
       CFLAGS="-qhalt=i"
    fi

    if test "${pgi_compiler}" = "no" ; then
       # Try linking with a weak alias...
       AC_LINK_IFELSE([
         AC_LANG_PROGRAM([
            void __alias(int c) {}
            void alias(int c) __attribute__((alias("__alias")));],
         [alias(0)])],
         [ax_cv_alias_attribute="yes"],
         [ax_cv_alias_attribute="no"])
     else
        ax_cv_alias_attribute="no"
     fi

     # Restore original CFLAGS
     CFLAGS=${save_CFLAGS}
     ])

  # What was the result of the test?
  AS_IF([test "${ax_cv_alias_attribute}" = "yes"],
  [
    AC_DEFINE([HAVE_ALIAS_ATTRIBUTE], 1,
              [Define this if aliases may be created with __attribute__])
  ])
])

AC_DEFUN([AX_CHECK_UNUSED_ATTRIBUTE],
[
  # Test whether compiler accepts __attribute__ form of setting unused 
  AC_CACHE_CHECK([whether ${CC} accepts function __attribute__((unused))],
  [ax_cv_unused_attribute], [

    # We add -Werror if it's gcc to force an error exit if the weak attribute
    # isn't understood

    save_CFLAGS=${CFLAGS}
    
    if test "${GCC}" = "yes" ; then
       CFLAGS="-Werror"
    elif test "`basename ${CC}`" = "xlc" ; then
       CFLAGS="-qhalt=i"
    fi

    if test "${pgi_compiler}" = "no" ; then
       # Try linking with a weak alias...
       AC_LINK_IFELSE([
         AC_LANG_PROGRAM(
         [
            static char var __attribute__((unused));],
            [])],
         [ax_cv_unused_attribute="yes"],
         [ax_cv_unused_attribute="no"])
     else
        ax_cv_unused_attribute="no"
     fi

    # Restore original CFLAGS
    CFLAGS=${save_CFLAGS}
  ])

  # What was the result of the test?
  AS_IF([test "${ax_cv_unused_attribute}" = "yes"],
  [
    AC_DEFINE([HAVE_UNUSED_ATTRIBUTE], 1,
              [Define this if variables/functions can be marked as unused])
  ])
])

AC_DEFUN([AX_CHECK_LOAD_BALANCING],
[
   AC_ARG_WITH(load-balancing,
   AC_HELP_STRING(
      [--with-load-balancing@<:@=DIR@:>@],
      [specify where to find "load balancing" libraries and includes]
      ),
      [lb_path="$withval"],
      [lb_path="none"] dnl List of possible default paths
   )
   if test "${lb_path}" != "none" ; then
      AC_MSG_CHECKING([for load-balancing installation])
      if test -r "${lb_path}/include/MPI_interface.h" ; then
         AC_MSG_RESULT([$lb_path])
         LOAD_BALANCING_HOME=${lb_path}
         AC_SUBST([LOAD_BALANCING_HOME])
         lb_found="yes"
      else
         AC_MSG_ERROR([load balancing headers not found])
         lb_found="no"
      fi
   fi
   AM_CONDITIONAL(GENERATE_LOAD_BALANCING, test "${lb_found}" = "yes" )
])

AC_DEFUN([AX_OFF_T_64BIT],
[
	AC_MSG_CHECKING([how to get 64-bit off_t])
	if test "${OperatingSystem}" = "linux" ; then
		AC_DEFINE([_FILE_OFFSET_BITS],[64],[Define the bits for the off_t structure])
		AC_MSG_RESULT([define _FILE_OFFSET_BITS=64])
	elif test "${OperatingSystem}" = "freebsd" ; then
		AC_MSG_RESULT([nothing required])
	else
		AC_MSG_RESULT([unknown])
	fi
])

AC_DEFUN([AX_CHECK_PROC_CPUINFO],
[
	AC_MSG_CHECKING(for /proc/cpuinfo)
	if test -r /proc/cpuinfo ; then
		AC_MSG_RESULT([found])
		AC_DEFINE([HAVE_PROC_CPUINFO], 1, [Define to 1 the OS has /proc/cpuinfo])
	else
		AC_MSG_RESULT([not found])
	fi
])

AC_DEFUN([AX_CHECK_PROC_MEMINFO],
[
	AC_MSG_CHECKING(for /proc/meminfo)
	if test -r /proc/meminfo ; then
		AC_MSG_RESULT([found])
		AC_DEFINE([HAVE_PROC_MEMINFO], 1, [Define to 1 the OS has /proc/meminfo])
	else
		AC_MSG_RESULT([not found])
	fi
])
