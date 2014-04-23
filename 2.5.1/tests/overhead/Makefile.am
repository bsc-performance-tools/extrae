include $(top_srcdir)/PATHS

EXTRA_DIST = \
 posix_clock.c \
 ia32_rdtsc_clock.c \
 extrae_eventandcounters.c \
 extrae_event.c \
 extrae_nevent4.c \
 extrae_get_caller1.c \
 extrae_get_caller6.c \
 extrae_trace_callers.c \
 extrae_user_function.c \
 papi_read1.c \
 papi_read4.c \
 Makefile.tests.overhead \
 run_overhead_tests.sh \
 extrae.xml

myFILES= \
 posix_clock.c \
 ia32_rdtsc_clock.c \
 extrae_eventandcounters.c \
 extrae_event.c \
 extrae_nevent4.c \
 extrae_get_caller1.c \
 extrae_get_caller6.c \
 extrae_trace_callers.c \
 extrae_user_function.c \
 papi_read1.c \
 papi_read4.c \
 run_overhead_tests.sh \
 extrae.xml

bin_PROGRAMS = extrae-header

extrae_header_SOURCES = extrae-header.c
extrae_header_CFLAGS = -g -O -Wall -I ../.. -I ../../src/common
extrae_header_LDFLAGS = -g -O
if HAVE_PAPI
extrae_header_CFLAGS += -DPAPI_HOME='"$(PAPI_HOME)"' @PAPI_CFLAGS@
extrae_header_LDFLAGS += -L@PAPI_SHAREDLIBSDIR@ -R@PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
if HAVE_UNWIND
extrae_header_CFLAGS += -DUNWIND_HOME='"$(UNWIND_HOME)"' @UNWIND_CFLAGS@
extrae_header_LDFLAGS += -L@UNWIND_SHAREDLIBSDIR@ -R@UNWIND_SHAREDLIBSDIR@ @UNWIND_LIBS@
endif
if IS_BG_MACHINE
extrae_header_LDFLAGS += -all-static
endif

install-data-hook:
	$(MKDIR_P) $(DESTDIR)$(datadir)/tests/overhead
	cp $(myFILES) $(DESTDIR)$(datadir)/tests/overhead
	cp Makefile.tests.overhead $(DESTDIR)$(datadir)/tests/overhead/Makefile
	$(BASE_DIR)/substitute $(SED) "@sub_EXTRAE_HOME@" "$(prefix)" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
	$(BASE_DIR)/substitute $(SED) "@sub_PAPI_SHAREDLIBSDIR@" "$(PAPI_SHAREDLIBSDIR)" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
	$(BASE_DIR)/substitute $(SED) "@sub_UNWIND_SHAREDLIBSDIR@" "$(UNWIND_SHAREDLIBSDIR)" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
if HAVE_PAPI
	$(BASE_DIR)/substitute $(SED) "@sub_COUNTERS_OVERHEAD_TESTS@" "./extrae_eventandcounters ./papi_read1 ./papi_read4" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
endif
if HAVE_UNWIND
	$(BASE_DIR)/substitute $(SED) "@sub_CALLERS_OVERHEAD_TESTS@" "./extrae_user_function ./extrae_get_caller1 ./extrae_get_caller6 ./extrae_trace_callers" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
endif

