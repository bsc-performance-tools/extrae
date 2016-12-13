
EXTRA_DIST = configured.sh extrae.sh

VARS_SCRIPT = extrae-vars.sh

noinst_SCRIPTS = $(VARS_SCRIPT)

$(VARS_SCRIPT): 
	echo "Configure the shell scripts (SH)"
	echo "# Extrae dependences" > $(VARS_SCRIPT)
	echo "export EXTRAE_MPI_HOME=${MPI_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_MPI_LIBSDIR=${MPI_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_PAPI_HOME=${PAPI_HOME}" >> $(VARS_SCRIPT) 
	echo "export EXTRAE_PAPI_LIBSDIR=${PAPI_LIBSDIR}" >> $(VARS_SCRIPT) 
	echo "export EXTRAE_LIBXML2_HOME=${XML2_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_LIBXML2_LIBSDIR=${XML2_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_DYNINST_HOME=${DYNINST_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_DYNINST_LIBSDIR=${DYNINST_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_DWARF_HOME=${DWARF_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_DWARF_LIBSDIR=${DWARF_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_ELF_HOME=${ELF_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_ELF_LIBSDIR=${ELF_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_LIBERTY_HOME=${LIBERTY_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_LIBERTY_LIBSDIR=${LIBERTY_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_BFD_HOME=${BFD_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_BFD_LIBSDIR=${BFD_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_UNWIND_HOME=${UNWIND_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_UNWIND_LIBSDIR=${UNWIND_LIBSDIR}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_BOOST_HOME=${BOOST_HOME}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_BOOST_LIBSDIR=${BOOST_HOME}/lib" >> $(VARS_SCRIPT)
	echo "export EXTRAE_CC=${CC}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_CFLAGS=\"${CFLAGS}\"" >> $(VARS_SCRIPT)
	echo "export EXTRAE_CXX=${CXX}" >> $(VARS_SCRIPT)
	echo "export EXTRAE_CXXFLAGS=\"${CXXFLAGS}\"" >> $(VARS_SCRIPT)
if HAVE_ONLINE
	echo "export EXTRAE_ONLINE_SOURCES=$(abs_builddir)/extrae_online.sh" >> $(VARS_SCRIPT)
endif
if INSTRUMENT_IO
	echo "export EXTRAE_INSTRUMENT_IO=1" >> $(VARS_SCRIPT)
endif
if INSTRUMENT_DYNAMIC_MEMORY
	echo "export EXTRAE_INSTRUMENT_DYNAMIC_MEMORY=1" >> $(VARS_SCRIPT)
endif

mkdirSHscripts:
	$(MKDIR_P) $(DESTDIR)$(sysconfdir)
	rm -fr $(DESTDIR)$(sysconfdir)/extrae-vars.sh $(DESTDIR)$(sysconfdir)/extrae-vars.csh

SHscripts:
	cp $(VARS_SCRIPT) $(DESTDIR)$(sysconfdir)
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae-vars.sh
	cp $(top_srcdir)/etc/extrae.sh $(DESTDIR)$(sysconfdir)
	chmod 0644 $(DESTDIR)$(sysconfdir)/extrae.sh

CONFIGUREDscript:
	echo "Preparing configured script for SH"
	echo "export PACKAGE_NAME=${PACKAGE_NAME}" >> $(VARS_SCRIPT)
	echo "export PACKAGE_BUGREPORT=${PACKAGE_BUGREPORT}" >> $(VARS_SCRIPT)
	echo "export CONFIGURE_LINE=\"`head -7 ${top_builddir}/config.log | tail -1`\"" >> $(VARS_SCRIPT)
	cp $(top_srcdir)/etc/configured.sh $(DESTDIR)$(sysconfdir)
	chmod 0755 $(DESTDIR)$(sysconfdir)/configured.sh

install-data-hook: mkdirSHscripts CONFIGUREDscript SHscripts
	$(top_srcdir)/substitute $(SED) "export EXTRAE_ONLINE_SOURCES=$(abs_builddir)/extrae_online.sh" "export EXTRAE_ONLINE_SOURCES=$(DESTDIR)$(sysconfdir)/extrae_online.sh" $(VARS_SCRIPT)

uninstall-local:
	rm $(DESTDIR)/$(sysconfdir)/extrae-vars.sh \
	   $(DESTDIR)$(sysconfdir)/extrae.sh \
	   $(DESTDIR)$(sysconfdir)/configured.sh
	rmdir --ignore-fail-on-non-empty $(DESTDIR)/$(sysconfdir)
