include_HEADERS = extrae_user_events.h extrae_types.h extrae_internals.h extraef_user_events.h

if HAVE_FC
noinst_PROGRAMS = extrae_module.mod
extrae_module_mod_SOURCES = extrae_module.f90
extrae_module.mod: extrae_module.f90
	${FC} -c $^ -o $@
endif

# install-exec-hook is executed before the regular objectives are performed
# install-data-hook is executed after the regular objectives are performed
#

install-data-hook:
	cd $(DESTDIR)$(includedir); $(LN_S) -f extrae_user_events.h mpitrace_user_events.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extrae_user_events.h extrae.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extraef_user_events.h mpitracef_user_events.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extraef_user_events.h extraef.h
if HAVE_FC
	cp extrae_module.mod $(DESTDIR)$(includedir)
endif

uninstall-local:
	rm $(DESTDIR)$(includedir)/mpitrace_user_events.h \
	   $(DESTDIR)$(includedir)/extrae.h \
	   $(DESTDIR)$(includedir)/mpitracef_user_events.h \
	   $(DESTDIR)$(includedir)/extraef.h
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(includedir)
