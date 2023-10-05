include_HEADERS = extrae_user_events.h extrae_types.h extrae_internals.h extraef_user_events.h extrae_module.f90

EXTRAE_VERSION_HEADER = extrae_version.h

EXTRA_DIST = $(EXTRAE_VERSION_HEADER).in Makefile.extrae_module

# install-exec-hook is executed before the regular objectives are performed
# install-data-hook is executed after the regular objectives are performed

install-data-hook:
	cd $(DESTDIR)$(includedir); $(LN_S) -f extrae_user_events.h mpitrace_user_events.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extrae_user_events.h extrae.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extraef_user_events.h mpitracef_user_events.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extraef_user_events.h extraef.h
if HAVE_FC
	cp $(top_srcdir)/include/Makefile.extrae_module $(DESTDIR)$(includedir)
	$(top_srcdir)/substitute $(SED) "@sub_PREFIXDIR@" "${prefix}" $(DESTDIR)$(includedir)/Makefile.extrae_module
	$(top_srcdir)/substitute $(SED) "@sub_FC@" "`which ${FC}`" $(DESTDIR)$(DESTDIR)$(includedir)/Makefile.extrae_module
endif
	cp $(EXTRAE_VERSION_HEADER).in $(DESTDIR)$(includedir)/$(EXTRAE_VERSION_HEADER)
	$(top_srcdir)/substitute $(SED) "@sub_EXTRAE_MAJOR@" "${EXTRAE_MAJOR}" $(DESTDIR)$(includedir)/$(EXTRAE_VERSION_HEADER)
	$(top_srcdir)/substitute $(SED) "@sub_EXTRAE_MINOR@" "${EXTRAE_MINOR}" $(DESTDIR)$(includedir)/$(EXTRAE_VERSION_HEADER)
	$(top_srcdir)/substitute $(SED) "@sub_EXTRAE_MICRO@" "${EXTRAE_MICRO}" $(DESTDIR)$(includedir)/$(EXTRAE_VERSION_HEADER)

uninstall-local:
	rm $(DESTDIR)$(includedir)/mpitrace_user_events.h \
	   $(DESTDIR)$(includedir)/extrae.h \
	   $(DESTDIR)$(includedir)/mpitracef_user_events.h \
	   $(DESTDIR)$(includedir)/extraef.h \
	   $(DESTDIR)$(includedir)/$(EXTRAE_VERSION_HEADER) \
	   $(DESTDIR)$(includedir)/extrae_module.* \
	   $(DESTDIR)$(includedir)/Makefile.extrae_module
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(includedir)
