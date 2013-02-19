
EXTRA_DIST = configured.sh extrae.csh extrae.sh

SH_SCRIPT = $(sysconfdir)/extrae-vars.sh
CSH_SCRIPT= $(sysconfdir)/extrae-vars.csh

mkdirSHscripts:
	$(mkdir_p) $(sysconfdir)
	touch $(sysconfdir)/license.dat
	rm -fr $(sysconfdir)/extrae-vars.sh $(sysconfdir)/extrae-vars.csh

SHscripts:
	echo "Configure the shell scripts (SH)"
	echo "export MPI_HOME=${MPI_HOME}" >> $(SH_SCRIPT)
	echo "export PACX_HOME=${PACX_HOME}" >> $(SH_SCRIPT)
	echo "export PAPI_HOME=${PAPI_HOME}" >> $(SH_SCRIPT) 
	echo "export LIBXML2_HOME=${XML2_HOME}" >> $(SH_SCRIPT)
	echo "export DYNINST_HOME=${DYNINST_HOME}" >> $(SH_SCRIPT)
	echo "export DWARF_HOME=${DWARF_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CC=${CC}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CFLAGS=\"${CFLAGS}\"" >> $(SH_SCRIPT)
	echo "export EXTRAE_CXX=${CXX}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CXXFLAGS=\"${CXXFLAGS}\"" >> $(SH_SCRIPT)
	echo "export UNWIND_HOME=${UNWIND_HOME}" >> $(SH_SCRIPT)
	chmod 0644 $(sysconfdir)/extrae-vars.sh
	cp extrae.sh $(sysconfdir)
	chmod 0644 $(sysconfdir)/extrae.sh

CSHscripts:
	echo "Configure the shell scripts (CSH)"
	echo "setenv MPI_HOME ${MPI_HOME}" >> $(CSH_SCRIPT)
	echo "setenv PACX_HOME ${PACX_HOME}" >> $(CSH_SCRIPT)
	echo "setenv PAPI_HOME ${PAPI_HOME}" >> $(CSH_SCRIPT)
	echo "setenv LIBXML2_HOME ${XML2_HOME}" >> $(CSH_SCRIPT)
	echo "setenv DYNINST_HOME ${DYNINST_HOME}" >> $(CSH_SCRIPT)
	echo "setenv DWARF_HOME ${DWARF_HOME}" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CC ${CC}" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CFLAGS \"${CFLAGS}\"" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CXX ${CXX}" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CXXFLAGS \"${CXXFLAGS}\"" >> $(CSH_SCRIPT)
	echo "setenv UNWIND_HOME ${UNWIND_HOME}" >> $(CSH_SCRIPT)
	chmod 0644 $(sysconfdir)/extrae-vars.csh
	cp extrae.csh $(sysconfdir)
	chmod 0644 $(sysconfdir)/extrae.csh

CONFIGUREDscript:
	echo "Preparing configured script for SH"
	echo "export PACKAGE_NAME=${PACKAGE_NAME}" >> $(SH_SCRIPT)
	echo "export PACKAGE_BUGREPORT=${PACKAGE_BUGREPORT}" >> $(SH_SCRIPT)
	echo "export CONFIGURE_LINE=\"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(SH_SCRIPT)
	echo "Preparing configured script for CSH"
	echo "setenv PACKAGE_NAME ${PACKAGE_NAME}" >> $(CSH_SCRIPT)
	echo "setenv PACKAGE_BUGREPORT ${PACKAGE_BUGREPORT}" >> $(CSH_SCRIPT)
	echo "setenv CONFIGURE_LINE \"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(CSH_SCRIPT)
	cp configured.sh $(sysconfdir)
	chmod 0755 $(sysconfdir)/configured.sh

install-data-hook: mkdirSHscripts SHscripts CSHscripts CONFIGUREDscript

uninstall-local:
	rm -f $(sysconfdir)/license.dat

