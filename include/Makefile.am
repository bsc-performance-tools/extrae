include_HEADERS = extrae_user_events.h extrae_types.h extrae_internals.h extraef_user_events.h extrae_module.f90

if IS_CELL_MACHINE
include_HEADERS += spu_trace.h ppu_trace_sdk1.h ppu_trace_sdk2.h
endif

#
# install-exec-hook is executed before the regular objectives are performed
# install-data-hook is executed after the regular objectives are performed
#

install-data-hook:
	cp $(COMMON_INC)/extrae_version.h $(DESTDIR)$(includedir)
	cd $(DESTDIR)$(includedir); $(LN_S) -f extrae_user_events.h mpitrace_user_events.h
	cd $(DESTDIR)$(includedir); $(LN_S) -f extraef_user_events.h mpitracef_user_events.h
if CELL_SDK_1
	cd $(DESTDIR)$(includedir); $(LN_S) -f ppu_trace_sdk1.h ppu_trace.h
else
if CELL_SDK_2
	cd $(DESTDIR)$(includedir); $(LN_S) -f ppu_trace_sdk2.h ppu_trace.h
endif
endif
