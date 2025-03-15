# AX_PROG_XML2
# -----------
# Check for libxml2 installation
AC_DEFUN([AX_PROG_XML2],
[
   AX_FLAGS_SAVE()

   AC_ARG_WITH(xml-prefix,
      AC_HELP_STRING(
         [--with-xml-prefix@<:@=DIR@:>@],
         [specify where to find libxml2 libraries and includes (deprecated, use --with-xml)]
      ),
      [xml_paths="${withval}"],
      [xml_paths="/usr/local /usr"] dnl List of possible default paths
   )

   AC_ARG_WITH(xml,
      AC_HELP_STRING(
         [--with-xml@<:@=DIR@:>@],
         [specify where to find libxml2 libraries and includes]
      ),
      [xml_paths="${withval}"]
      dnl [xml_paths="/usr/local /usr"] dnl List of possible default not given here, taken from --with-xml-prefix
      dnl                               dnl Activate this again when --with-xml-prefix is removed
   )

   AX_FIND_INSTALLATION([XML2], [${xml_paths}], [xml2-config], [], [], [], [xml2], [], [], [])
   if test "$XML2_INSTALLED" = "yes" ; then

      min_xml_version=ifelse([$1], ,2.0.0, [$1])
      AC_MSG_CHECKING(for libxml2 version >= $min_xml_version)
      min_xml_major_version=`echo ${min_xml_version} | cut -d. -f1`
      min_xml_minor_version=`echo ${min_xml_version} | cut -d. -f2`
      min_xml_micro_version=`echo ${min_xml_version} | cut -d. -f3`

      xml_config_major_version=`${XML2_BIN_xml2_config} --version | cut -d. -f1`
      xml_config_minor_version=`${XML2_BIN_xml2_config} --version | cut -d. -f2`
      xml_config_micro_version=`${XML2_BIN_xml2_config} --version | cut -d. -f3`

      if ((xml_config_major_version > min_xml_major_version)) ||
         ((xml_config_major_version == min_xml_major_version && xml_config_minor_version > min_xml_minor_version)) ||
         ((xml_config_major_version == major && xml_config_minor_version == min_xml_minor_version && xml_config_micro_version >= min_xml_micro_version)); then
         AC_MSG_RESULT([yes ($xml_config_major_version.$xml_config_minor_version.$xml_config_micro_version)])
      else
         AC_MSG_RESULT([no ($xml_config_major_version.$xml_config_minor_version.$xml_config_micro_version)])
         XML2_INSTALLED="no"
      fi

      XML2_CFLAGS="${XML2_CFLAGS} -I${XML2_INCLUDES}/libxml2"
      XML2_CPPFLAGS="${XML2_CPPFLAGS} -I${XML2_INCLUDES}/libxml2"
      XML2_CXXFLAGS="${XML2_CXXFLAGS} -I${XML2_INCLUDES}/libxml2"
      AC_SUBST(XML2_CFLAGS)
      AC_SUBST(XML2_CPPFLAGS)
      AC_SUBST(XML2_CXXFLAGS)

      CFLAGS=${XML2_CFLAGS}
      AC_CHECK_HEADERS([libxml/parser.h libxml/xmlmemory.h], [], [$XML2_INSTALLED="no"])
   fi
   AX_FLAGS_RESTORE()

   AM_CONDITIONAL(HAVE_XML2, test "${XML2_INSTALLED}" = "yes")
   if test "$XML2_INSTALLED" = "yes" ; then
      AC_DEFINE([HAVE_XML2], [1], [Defined if libxml2 is available])
   fi
])