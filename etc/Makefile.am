
EXTRA_DIST = configured.sh extrae.csh extrae.sh

SH_SCRIPT = $(DESTDIR)$(sysconfdir)/extrae-vars.sh
CSH_SCRIPT= $(DESTDIR)$(sysconfdir)/extrae-vars.csh

mkdirSHscripts:
	$(MKDIR_P) $(DESTDIR)$(sysconfdir)
	rm -fr $(DESTDIR)$(sysconfdir)/extrae-vars.sh $(DESTDIR)$(sysconfdir)/extrae-vars.csh

SHscripts:
	echo "Configure the shell scripts (SH)"
	echo "export MPI_HOME=${MPI_HOME}" >> $(SH_SCRIPT)
	echo "export PACX_HOME=${PACX_HOME}" >> $(SH_SCRIPT)
	echo "export PAPI_HOME=${PAPI_HOME}" >> $(SH_SCRIPT) 
	echo "export LIBXML2_HOME=${XML2_HOME}" >> $(SH_SCRIPT)
	echo "export DYNINST_HOME=${DYNINST_HOME}" >> $(SH_SCRIPT)
	echo "export DYNINST_LIBSDIR=${DYNINST_LIBSDIR}" >> $(SH_SCRIPT)
	echo "export DWARF_HOME=${DWARF_HOME}" >> $(SH_SCRIPT)
	echo "export ELF_HOME=${ELF_HOME}" >> $(SH_SCRIPT)
	echo "export LIBERTY_HOME=${LIBERTY_HOME}" >> $(SH_SCRIPT)
	echo "export BFD_HOME=${BFD_HOME}" >> $(SH_SCRIPT)
	echo "export UNWIND_HOME=${UNWIND_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CC=${CC}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CFLAGS=\"${CFLAGS}\"" >> $(SH_SCRIPT)
	echo "export EXTRAE_CXX=${CXX}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CXXFLAGS=\"${CXXFLAGS}\"" >> $(SH_SCRIPT)
	echo "export EXTRAE_SVN_REVISION=`cat ../SVN-revision`" >> $(SH_SCRIPT)
	echo "export EXTRAE_SVN_BRANCH=`cat ../SVN-branch`" >> $(SH_SCRIPT)
if HAVE_ONLINE
	echo "export EXTRAE_ONLINE_SOURCES=$(DESTDIR)$(sysconfdir)/online_env.sh" >> $(SH_SCRIPT)
endif
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae-vars.sh
	cp $(top_srcdir)/etc/extrae.sh $(DESTDIR)$(sysconfdir)
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae.sh

CSHscripts:
	echo "Configure the shell scripts (CSH)"
	echo "setenv MPI_HOME ${MPI_HOME}" >> $(CSH_SCRIPT)
	echo "setenv PACX_HOME ${PACX_HOME}" >> $(CSH_SCRIPT)
	echo "setenv PAPI_HOME ${PAPI_HOME}" >> $(CSH_SCRIPT)
	echo "setenv LIBXML2_HOME ${XML2_HOME}" >> $(CSH_SCRIPT)
	echo "setenv DYNINST_HOME ${DYNINST_HOME}" >> $(CSH_SCRIPT)
	echo "setenv DYNINST_LIBSDIR ${DYNINST_LIBSDIR}" >> $(CSH_SCRIPT)
	echo "setenv DWARF_HOME ${DWARF_HOME}" >> $(CSH_SCRIPT)
	echo "setenv ELF_HOME ${ELF_HOME}" >> $(CSH_SCRIPT)
	echo "setenv LIBERTY_HOME ${LIBERTY_HOME}" >> $(CSH_SCRIPT)
	echo "setenv BFD_HOME ${BFD_HOME}" >> $(CSH_SCRIPT)
	echo "setenv UNWIND_HOME ${UNWIND_HOME}" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CC ${CC}" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CFLAGS \"${CFLAGS}\"" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CXX ${CXX}" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_CXXFLAGS \"${CXXFLAGS}\"" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_SVN_REVISION `cat ../SVN-revision`" >> $(CSH_SCRIPT)
	echo "setenv EXTRAE_SVN_BRANCH `cat ../SVN-branch`" >> $(CSH_SCRIPT)
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae-vars.csh
	cp $(top_srcdir)/etc/extrae.csh $(DESTDIR)$(sysconfdir)
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae.csh

CONFIGUREDscript:
	echo "Preparing configured script for SH"
	echo "export PACKAGE_NAME=${PACKAGE_NAME}" >> $(SH_SCRIPT)
	echo "export PACKAGE_BUGREPORT=${PACKAGE_BUGREPORT}" >> $(SH_SCRIPT)
	echo "export CONFIGURE_LINE=\"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(SH_SCRIPT)
	echo "Preparing configured script for CSH"
	echo "setenv PACKAGE_NAME ${PACKAGE_NAME}" >> $(CSH_SCRIPT)
	echo "setenv PACKAGE_BUGREPORT ${PACKAGE_BUGREPORT}" >> $(CSH_SCRIPT)
	echo "setenv CONFIGURE_LINE \"`head -7 ${top_srcdir}/config.log | tail -1`\"" >> $(CSH_SCRIPT)
	cp $(top_srcdir)/etc/configured.sh $(DESTDIR)$(sysconfdir)
	chmod 0755 $(DESTDIR)$(sysconfdir)/configured.sh

install-data-hook: mkdirSHscripts SHscripts CSHscripts CONFIGUREDscript

