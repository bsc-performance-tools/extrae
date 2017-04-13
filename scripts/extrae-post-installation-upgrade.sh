#!/bin/bash

function look_for_var
{
	var=${1}
	file=${2}

	# Look for export x=y or for x=y
	while read line
	do
		tmp=`echo ${line} | cut -f 1 -d '='`
		if test "${tmp}" = "export ${var}" -o "${tmp}" = "${var}"; then
      look_for_var_res=`echo ${line} | cut -f 2- -d '='`
			return
		fi
	done < ${file}
}

function replace_var_dir
{
	look_for_var ${1} ${2}
	res=${look_for_var_res}

	echo "Current value for ${1} is ${res}. Are you happy with it? (yes/no)"
	while read happy
	do
		if test "${happy}" = "yes" -o "${happy}" = "y" -o "${happy}" = "no" -o "${happy}" = "n" ; then
			break
		else
			echo "Are you happy with it? (yes/no)"
		fi
	done

	if test "${happy}" = "no" -o "${happy}" = "n" ; then
		echo "Which shall be the value for ${1}?"
		while read replace_var_res
		do
			if test "${replace_var_res}" != "" ; then
				if test ! -d "${replace_var_res}" ; then
					echo "Error! ${1} should point to a directory, but ${replace_var_res} is not. Please, give a new value for ${1}"
				else
					break
				fi
			else
				break
			fi
		done
	else
		replace_var_res=${res}
	fi
}

function generate_extrae_vars_csh
{
	echo "Generating a newer ${EXTRAE_HOME}/etc/extrae-vars.csh"

	rm -f ${EXTRAE_HOME}/etc/extrae-vars.csh ; touch ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv MPI_HOME ${new_MPI_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv PAPI_HOME ${new_PAPI_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv LIBXML2_HOME ${new_LIBXML2_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv DYNINST_HOME ${new_DYNINST_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv DWARF_HOME ${new_DWARF_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv ELF_HOME ${new_ELF_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv LIBERTY_HOME ${new_LIBERTY_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv BFD_HOME ${new_BFD_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv UNWIND_HOME ${new_UNWIND_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv EXTRAE_CC ${new_EXTRAE_CC}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv EXTRAE_CFLAGS ${new_EXTRAE_CFLAGS}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv EXTRAE_CXX ${new_EXTRAE_CXX}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv EXTRAE_CXXFLAGS ${new_EXTRAE_CXXFLAGS}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv PACKAGE_NAME ${new_PACKAGE_NAME}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv PACKAGE_BUGREPORT ${new_PACKAGE_BUGREPORT}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
	echo "setenv CONFIGURE_LINE ${new_CONFIGURE_LINE}" >> ${EXTRAE_HOME}/etc/extrae-vars.csh
}


function generate_extrae_vars_sh
{
	echo "Generating a newer ${EXTRAE_HOME}/etc/extrae-vars.sh"

	rm -f ${EXTRAE_HOME}/etc/extrae-vars.sh ; touch ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export MPI_HOME=${new_MPI_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export PAPI_HOME=${new_PAPI_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export LIBXML2_HOME=${new_LIBXML2_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export DYNINST_HOME=${new_DYNINST_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export DWARF_HOME=${new_DWARF_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export ELF_HOME=${new_ELF_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export LIBERTY_HOME=${new_LIBERTY_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export BFD_HOME=${new_BFD_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export UNWIND_HOME=${new_UNWIND_HOME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export EXTRAE_CC=${new_EXTRAE_CC}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export EXTRAE_CFLAGS=${new_EXTRAE_CFLAGS}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export EXTRAE_CXX=${new_EXTRAE_CXX}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export EXTRAE_CXXFLAGS=${new_EXTRAE_CXXFLAGS}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export PACKAGE_NAME=${new_PACKAGE_NAME}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export PACKAGE_BUGREPORT=${new_PACKAGE_BUGREPORT}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
	echo "export CONFIGURE_LINE=${new_CONFIGURE_LINE}" >> ${EXTRAE_HOME}/etc/extrae-vars.sh
}

function generate_extrae_Makefile_inc
{
	look_for_var POSIX_CLOCK_LIBS ${EXTRAE_HOME}/share/example/Makefile.inc
	new_POSIX_CLOCK_LIBS=${look_for_var_res}
	look_for_var LIBEXECINFO_LIBS ${EXTRAE_HOME}/share/example/Makefile.inc
	new_LIBEXECINFO_LIBS=${look_for_var_res}
	look_for_var XML2_LDFLAGS ${EXTRAE_HOME}/share/example/Makefile.inc
	new_XML2_LDFLAGS=${look_for_var_res}
	look_for_var BFD_LIBS ${EXTRAE_HOME}/share/example/Makefile.inc
	new_BFD_LIBS=${look_for_var_res}
	look_for_var CUPTI_LIBS ${EXTRAE_HOME}/share/example/Makefile.inc
	new_CUPTI_LIBS=${look_for_var_res}

	echo "Generating a newer ${EXTRAE_HOME}/share/example/Makefile.inc"

	rm -f ${EXTRAE_HOME}/share/example/Makefile.inc ; touch ${EXTRAE_HOME}/share/example/Makefile.inc

	echo "EXTRAE_HOME=${EXTRAE_HOME}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "MPI_HOME=${new_MPI_HOME}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "PAPI_HOME=${new_PAPI_HOME}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "XML2_HOME=${new_LIBXML2_HOME}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	if test -x ${new_LIBXML2_HOME}/bin/xml2-config ; then
		tmp=`${new_LIBXML2_HOME}/bin/xml2-config --libs`
		echo "XML2_LDFLAGS=${new_XML2_LDFLAGS}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
		echo "XML2_LIBS=${tmp}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	else
		echo "Severe Warning! Cannot locate xml2-config in \$XML2_HOME/bin. Check for XML2_LDFLAGS and XML2_LIBS in ${EXTRAE_HOME}/share/example/Makefile.inc"
		echo "XML2_LDFLAGS=-L$(XML2_LIBS)/lib # You may want to edit this" >> ${EXTRAE_HOME}/share/example/Makefile.inc
		echo "XML2_LIBS=<edit this!>" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	fi
	echo "UNWIND_HOME=${new_UNWIND_HOME}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "BFD_HOME=${new_BFD_HOME}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "LIBERTY_HOME=${new_LIBERTY_HOME}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "BFD_LIBS=${new_BFD_LIBS}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "UNWIND_LIBS=-L\$(PAPI_HOME)/lib" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "PAPI_LIBS=-L\$(PAPI_HOME)/lib" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "CUPTI_LIBS=${new_CUPTI_LIBS}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "POSIX_CLOCK_LIBS=${new_POSIX_CLOCK_LIBS}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
	echo "LIBEXEC_INFO_LIBS=${new_LIBEXECINFO_LIBS}" >> ${EXTRAE_HOME}/share/example/Makefile.inc
}

###
###
### SCRIPT ENTRY POINT
###
###

if test "${EXTRAE_HOME}" = "" ; then
 echo Error! You have to point the location of the Extrae package through EXTRAE_HOME environment variable.
 exit -1
fi

if test ! -d ${EXTRAE_HOME} ; then
 echo Error! EXTRAE_HOME environment variable is not pointing to a directory.
 exit -1
fi

echo
echo "**** Welcome to Extrae post-installation script"
echo 
echo "This script is intended to be used after uncompressing a binary installation of Extrae. It allows you to modify some values given at configure time in order to make Extrae run in your computer. Note, however, that this script does not change the binary distributed contents so you can't:"
echo "  * change the implementation of MPI - i.e. you can't use a package compiled for OpenMPI with MPICH or any other MPI implementation,"
echo "  * you can't add or remove PAPI support"
echo "  * you can't use 32 bit packages on 64 bit systems - or viceversa"
echo
echo "You're about to configure the package installed in: ${EXTRAE_HOME}"
echo

##
## Work first on the user setteable values
##

replace_var_dir MPI_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_MPI_HOME=${replace_var_res} ; echo
replace_var_dir PAPI_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_PAPI_HOME=${replace_var_res} ; echo
replace_var_dir UNWIND_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_UNWIND_HOME=${replace_var_res} ; echo
replace_var_dir LIBXML2_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_LIBXML2_HOME=${replace_var_res} ; echo
replace_var_dir DYNINST_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_DYNINST_HOME=${replace_var_res} ; echo
replace_var_dir DWARF_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_DWARF_HOME=${replace_var_res} ; echo
replace_var_dir ELF_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_ELF_HOME=${replace_var_res} ; echo
replace_var_dir LIBERTY_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_LIBERTY_HOME=${replace_var_res} ; echo
replace_var_dir BFD_HOME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_BFD_HOME=${replace_var_res} ; echo

##
## Now read those vars which can't be modified
##

look_for_var EXTRAE_CC ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_EXTRAE_CC=${look_for_var_res}
look_for_var EXTRAE_CFLAGS ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_EXTRAE_CFLAGS=${look_for_var_res}
look_for_var EXTRAE_CXX ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_EXTRAE_CXX=${look_for_var_res}
look_for_var EXTRAE_CXXFLAGS ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_EXTRAE_CXXFLAGS=${look_for_var_res}
look_for_var PACKAGE_NAME ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_PACKAGE_NAME=${look_for_var_res}
look_for_var PACKAGE_BUGREPORT ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_PACKAGE_BUGREPORT=${look_for_var_res}
look_for_var CONFIGURE_LINE ${EXTRAE_HOME}/etc/extrae-vars.sh ; new_CONFIGURE_LINE=${look_for_var_res}

##
## Regenerate the files that should be modified
##

echo Extrae post configuration summary:
echo --
echo MPI_HOME=${new_MPI_HOME}
echo PAPI_HOME=${new_UNWIND_HOME}
echo LIBXML2_HOME=${new_LIBXML2_HOME}
echo DYNINST_HOME=${new_DYNINST_HOME}
echo DWARF_HOME=${new_DWARF_HOME}
echo ELF_HOME=${new_ELF_HOME}
echo LIBERTY_HOME=${new_LIBERTY_HOME}
echo BFD_HOME=${new_BFD_HOME}
echo
echo "Are you happy with your selections? (yes/no)"
while read happy
do
	if test "${happy}" = "yes" -o "${happy}" = "y" -o "${happy}" = "no" -o "${happy}" = "n" ; then
		break
	else
		echo "Are you happy with your selections? (yes/no)"
		fi
done

if test "${happy}" = "y" -o "${happy}" = "yes"; then
  generate_extrae_vars_csh
  generate_extrae_vars_sh
  generate_extrae_Makefile_inc
else
  echo Dismissing changes
fi

