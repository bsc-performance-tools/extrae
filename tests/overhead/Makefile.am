include $(top_srcdir)/PATHS

EXTRA_DIST = \
 extrae_eventandcounters.c \
 extrae_event.c \
 extrae_get_caller1.c \
 extrae_get_caller4.c \
 extrae_trace_callers.c \
 extrae_user_function.c \
 papi_read1.c \
 papi_read4.c \
 Makefile.tests.overhead \
 run_overhead_tests.sh \
 extrae.xml

myFILES= \
 extrae_eventandcounters.c \
 extrae_event.c \
 extrae_get_caller1.c \
 extrae_get_caller4.c \
 extrae_trace_callers.c \
 extrae_user_function.c \
 papi_read1.c \
 papi_read4.c \
 run_overhead_tests.sh \
 extrae.xml

install-data-hook:
	$(mkdir_p) $(DESTDIR)$(datadir)/tests/overhead
	cp $(myFILES) $(DESTDIR)$(datadir)/tests/overhead
	cp Makefile.tests.overhead $(DESTDIR)$(datadir)/tests/overhead/Makefile
	$(BASE_DIR)/substitute $(SED) "@sub_EXTRAE_HOME@" "$(prefix)" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
	$(BASE_DIR)/substitute $(SED) "@sub_PAPI_SHAREDLIBSDIR@" "$(PAPI_SHAREDLIBSDIR)" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh

