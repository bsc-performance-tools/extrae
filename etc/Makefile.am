
mkdirSHscripts:
	$(mkdir_p) $(sysconfdir)
	touch $(sysconfdir)/license.dat
	rm -fr $(sysconfdir)/ompitrace-vars.sh $(sysconfdir)/ompitrace-vars.csh

SHscripts:
# Configure the shell scripts (SH)
	echo "export MPI_HOME=${MPI_HOME}" >> $(sysconfdir)/ompitrace-vars.sh
	echo "export PACX_HOME=${PACX_HOME}" >> $(sysconfdir)/ompitrace-vars.sh
	echo "export PAPI_HOME=${PAPI_HOME}" >> $(sysconfdir)/ompitrace-vars.sh
	echo "export LIBXML2_HOME=${XML2_HOME}" >> $(sysconfdir)/ompitrace-vars.sh
	echo "export DYNINST_HOME=${DYNINST_HOME}" >> $(sysconfdir)/ompitrace-vars.sh
	chmod 0644 $(sysconfdir)/ompitrace-vars.sh
	cp ompitrace.sh $(sysconfdir)
	chmod 0644 $(sysconfdir)/ompitrace.sh

CSHscripts:
# Configure the shell scripts (CSH)
	echo "setenv MPI_HOME ${MPI_HOME}" >> $(sysconfdir)/ompitrace-vars.csh
	echo "setenv PACX_HOME ${PACX_HOME}" >> $(sysconfdir)/ompitrace-vars.csh
	echo "setenv PAPI_HOME ${PAPI_HOME}" >> $(sysconfdir)/ompitrace-vars.csh
	echo "setenv LIBXML2_HOME ${XML2_HOME}" >> $(sysconfdir)/ompitrace-vars.csh
	echo "setenv DYNINST_HOME ${DYNINST_HOME}" >> $(sysconfdir)/ompitrace-vars.csh
	chmod 0644 $(sysconfdir)/ompitrace-vars.csh
	cp ompitrace.csh $(sysconfdir)
	chmod 0644 $(sysconfdir)/ompitrace.csh

CONFIGUREDscript:
	echo "setenv PACKAGE_NAME ${PACKAGE_NAME}" >> $(sysconfdir)/ompitrace-vars.csh
	echo "setenv PACKAGE_BUGREPORT ${PACKAGE_BUGREPORT}" >> $(sysconfdir)/ompitrace-vars.csh
	echo "setenv CONFIGURE_LINE \"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(sysconfdir)/ompitrace-vars.csh
	echo "export PACKAGE_NAME=${PACKAGE_NAME}" >> $(sysconfdir)/ompitrace-vars.sh
	echo "export PACKAGE_BUGREPORT=${PACKAGE_BUGREPORT}" >> $(sysconfdir)/ompitrace-vars.sh
	echo "export CONFIGURE_LINE=\"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(sysconfdir)/ompitrace-vars.sh
	cp configured.sh $(sysconfdir)
	chmod 0755 $(sysconfdir)/configured.sh

install-data-hook: mkdirSHscripts SHscripts CSHscripts CONFIGUREDscript

uninstall-local:
	rm -f $(sysconfdir)/license.dat

