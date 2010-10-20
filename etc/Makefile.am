
EXTRA_DIST = configured.sh extrae.csh extrae.sh

mkdirSHscripts:
	$(mkdir_p) $(sysconfdir)
	touch $(sysconfdir)/license.dat
	rm -fr $(sysconfdir)/extrae-vars.sh $(sysconfdir)/extrae-vars.csh

SHscripts:
# Configure the shell scripts (SH)
	echo "export MPI_HOME=${MPI_HOME}" >> $(sysconfdir)/extrae-vars.sh
	echo "export PACX_HOME=${PACX_HOME}" >> $(sysconfdir)/extrae-vars.sh
	echo "export PAPI_HOME=${PAPI_HOME}" >> $(sysconfdir)/extrae-vars.sh
	echo "export LIBXML2_HOME=${XML2_HOME}" >> $(sysconfdir)/extrae-vars.sh
	echo "export DYNINST_HOME=${DYNINST_HOME}" >> $(sysconfdir)/extrae-vars.sh
	echo "export DWARF_HOME=${DWARF_HOME}" >> $(sysconfdir)/extrae-vars.sh
	echo "export EXTRAE_CC=${CC}" >> $(sysconfdir)/extrae-vars.sh
	echo "export EXTRAE_CFLAGS=${CFLAGS}" >> $(sysconfdir)/extrae-vars.sh
	echo "export EXTRAE_CXX=${CXX}" >> $(sysconfdir)/extrae-vars.sh
	echo "export EXTRAE_CXXFLAGS=${CXXFLAGS}" >> $(sysconfdir)/extrae-vars.sh
	echo "export UNWIND_HOME=${UNWIND_HOME}" >> $(sysconfdir)/extrae-vars.sh
	chmod 0644 $(sysconfdir)/extrae-vars.sh
	cp extrae.sh $(sysconfdir)
	chmod 0644 $(sysconfdir)/extrae.sh

CSHscripts:
# Configure the shell scripts (CSH)
	echo "setenv MPI_HOME ${MPI_HOME}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv PACX_HOME ${PACX_HOME}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv PAPI_HOME ${PAPI_HOME}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv LIBXML2_HOME ${XML2_HOME}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv DYNINST_HOME ${DYNINST_HOME}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv DWARF_HOME ${DWARF_HOME}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv EXTRAE_CC ${CC}" >> $(sysconfdir)/extrae-vars.sh
	echo "setenv EXTRAE_CFLAGS ${CFLAGS}" >> $(sysconfdir)/extrae-vars.sh
	echo "setenv EXTRAE_CXX ${CXX}" >> $(sysconfdir)/extrae-vars.sh
	echo "setenv EXTRAE_CXXFLAGS ${CXXFLAGS}" >> $(sysconfdir)/extrae-vars.sh
	echo "setenv UNWIND_HOME ${UNWIND_HOME}" >> $(sysconfdir)/extrae-vars.sh
	chmod 0644 $(sysconfdir)/extrae-vars.csh
	cp extrae.csh $(sysconfdir)
	chmod 0644 $(sysconfdir)/extrae.csh

CONFIGUREDscript:
	echo "setenv PACKAGE_NAME ${PACKAGE_NAME}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv PACKAGE_BUGREPORT ${PACKAGE_BUGREPORT}" >> $(sysconfdir)/extrae-vars.csh
	echo "setenv CONFIGURE_LINE \"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(sysconfdir)/extrae-vars.csh
	echo "export PACKAGE_NAME=${PACKAGE_NAME}" >> $(sysconfdir)/extrae-vars.sh
	echo "export PACKAGE_BUGREPORT=${PACKAGE_BUGREPORT}" >> $(sysconfdir)/extrae-vars.sh
	echo "export CONFIGURE_LINE=\"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(sysconfdir)/extrae-vars.sh
	cp configured.sh $(sysconfdir)
	chmod 0755 $(sysconfdir)/configured.sh

install-data-hook: mkdirSHscripts SHscripts CSHscripts CONFIGUREDscript

uninstall-local:
	rm -f $(sysconfdir)/license.dat

