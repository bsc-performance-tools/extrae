
EXTRA_DIST = configured.sh extrae.sh

SH_SCRIPT = $(DESTDIR)$(sysconfdir)/extrae-vars.sh

mkdirSHscripts:
	$(MKDIR_P) $(DESTDIR)$(sysconfdir)
	rm -fr $(DESTDIR)$(sysconfdir)/extrae-vars.sh $(DESTDIR)$(sysconfdir)/extrae-vars.csh

SHscripts:
	echo "Configure the shell scripts (SH)"
	echo "export EXTRAE_MPI_HOME=${MPI_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_PAPI_HOME=${PAPI_HOME}" >> $(SH_SCRIPT) 
	echo "export EXTRAE_LIBXML2_HOME=${XML2_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_DYNINST_HOME=${DYNINST_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_DYNINST_LIBSDIR=${DYNINST_LIBSDIR}" >> $(SH_SCRIPT)
	echo "export EXTRAE_DWARF_HOME=${DWARF_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_ELF_HOME=${ELF_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_LIBERTY_HOME=${LIBERTY_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_BFD_HOME=${BFD_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_UNWIND_HOME=${UNWIND_HOME}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CC=${CC}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CFLAGS=\"${CFLAGS}\"" >> $(SH_SCRIPT)
	echo "export EXTRAE_CXX=${CXX}" >> $(SH_SCRIPT)
	echo "export EXTRAE_CXXFLAGS=\"${CXXFLAGS}\"" >> $(SH_SCRIPT)
	echo "export EXTRAE_SVN_REVISION=`cat ../SVN-revision`" >> $(SH_SCRIPT)
	echo "export EXTRAE_SVN_BRANCH=`cat ../SVN-branch`" >> $(SH_SCRIPT)
if HAVE_ONLINE
	echo "export EXTRAE_ONLINE_SOURCES=$(DESTDIR)$(sysconfdir)/extrae_online.sh" >> $(SH_SCRIPT)
endif
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae-vars.sh
	cp $(top_srcdir)/etc/extrae.sh $(DESTDIR)$(sysconfdir)
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae.sh

CONFIGUREDscript:
	echo "Preparing configured script for SH"
	echo "export PACKAGE_NAME=${PACKAGE_NAME}" >> $(SH_SCRIPT)
	echo "export PACKAGE_BUGREPORT=${PACKAGE_BUGREPORT}" >> $(SH_SCRIPT)
	echo "export CONFIGURE_LINE=\"`head -7 ${top_builddir}/config.log | tail -1`\"" >> $(SH_SCRIPT)
	cp $(top_srcdir)/etc/configured.sh $(DESTDIR)$(sysconfdir)
	chmod 0755 $(DESTDIR)$(sysconfdir)/configured.sh

install-data-hook: mkdirSHscripts SHscripts CONFIGUREDscript

