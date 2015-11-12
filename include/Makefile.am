include_HEADERS = extrae_user_events.h extrae_types.h extrae_internals.h extraef_user_events.h extrae_module.f90

# install-exec-hook is executed before the regular objectives are performed
# install-data-hook is executed after the regular objectives are performed
#

install-data-hook:
	cp $(COMMON_INC)/extrae_version.h $(DESTDIR)$(includedir)
	cd $(DESTDIR)$(includedir); $(LN_S) -f extrae_user_events.h mpitrace_user_events.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extrae_user_events.h extrae.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extraef_user_events.h mpitracef_user_events.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extraef_user_events.h extraef.h
