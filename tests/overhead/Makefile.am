include $(top_srcdir)/PATHS

EXTRA_DIST = \
 posix_clock.c \
 ia32_rdtsc_clock.c \
 ppc_clock.c \
 extrae_eventandcounters.c \
 extrae_event.c \
 extrae_nevent4.c \
 extrae_get_caller1.c \
 extrae_get_caller6.c \
 extrae_trace_callers.c \
 extrae_user_function.c \
 papi_read1.c \
 papi_read4.c \
 mutex.c \
 rwlock.c \
 Makefile.tests.overhead \
 run_overhead_tests.sh \
 extrae.xml \
 JavaEvent.java \
 JavaNEvent4.java \
 JavaFakeRoutine.java \
 extraej.xml \
 extraej_functionlist

myPATH="$(top_srcdir)/tests/overhead"

myFILES= \
 $(myPATH)/posix_clock.c \
 $(myPATH)/ia32_rdtsc_clock.c \
 $(myPATH)/ppc_clock.c \
 $(myPATH)/extrae_eventandcounters.c \
 $(myPATH)/extrae_event.c \
 $(myPATH)/extrae_nevent4.c \
 $(myPATH)/extrae_get_caller1.c \
 $(myPATH)/extrae_get_caller6.c \
 $(myPATH)/extrae_trace_callers.c \
 $(myPATH)/extrae_user_function.c \
 $(myPATH)/papi_read1.c \
 $(myPATH)/papi_read4.c \
 $(myPATH)/run_overhead_tests.sh \
 $(myPATH)/extrae.xml \
 $(myPATH)/JavaEvent.java \
 $(myPATH)/JavaNEvent4.java \
 $(myPATH)/JavaFakeRoutine.java \
 $(myPATH)/extraej.xml \
 $(myPATH)/extraej_functionlist

bin_PROGRAMS = extrae-header

extrae_header_SOURCES = extrae-header.c
extrae_header_CFLAGS = -g -O -I$(top_srcdir) -I$(COMMON_INC)
extrae_header_LDFLAGS = -g -O
if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  extrae_header_CFLAGS += -DPAPI_HOME='"$(PAPI_HOME)"' @PAPI_CFLAGS@
  extrae_header_LDFLAGS += @PAPI_LDFLAGS@ @PAPI_RPATH@ @PAPI_LIBS@
else
  extrae_header_CFLAGS += -DPAPI_HOME='"embedded"'
endif
endif
if HAVE_UNWIND
  extrae_header_CFLAGS += -DUNWIND_HOME='"$(UNWIND_HOME)"' @UNWIND_CFLAGS@
  extrae_header_LDFLAGS += @UNWIND_LDFLAGS@ @UNWIND_RPATH@ @UNWIND_LIBS@
endif
if IS_BG_MACHINE
 extrae_header_LDFLAGS += -all-static
endif

install-data-hook:
	$(MKDIR_P) $(DESTDIR)$(datadir)/tests/overhead
	cp $(myFILES) $(DESTDIR)$(datadir)/tests/overhead
	cp $(myPATH)/Makefile.tests.overhead $(DESTDIR)$(datadir)/tests/overhead/Makefile
	$(BASE_DIR)/substitute $(SED) "@sub_EXTRAE_HOME@" "$(DESTDIR)$(prefix)" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
if HAVE_PAPI
	$(BASE_DIR)/substitute $(SED) "@sub_COUNTERS_OVERHEAD_TESTS@" "./extrae_eventandcounters ./papi_read1 ./papi_read4" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
endif
if HAVE_UNWIND
	$(BASE_DIR)/substitute $(SED) "@sub_CALLERS_OVERHEAD_TESTS@" "./extrae_user_function ./extrae_get_caller1 ./extrae_get_caller6 ./extrae_trace_callers" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
endif
if WANT_JAVA
	$(top_srcdir)/substitute $(SED) "@sub_JAVA@" "`which ${JAVA}`" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
else
	$(top_srcdir)/substitute $(SED) "@sub_JAVA@" "" $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh
endif

uninstall-local:
	rm $(DESTDIR)$(datadir)/tests/overhead/Makefile \
	   $(DESTDIR)$(datadir)/tests/overhead/posix_clock.c \
	   $(DESTDIR)$(datadir)/tests/overhead/ia32_rdtsc_clock.c \
	   $(DESTDIR)$(datadir)/tests/overhead/ppc_clock.c \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae_eventandcounters.c \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae_event.c \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae_nevent4.c \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae_get_caller1.c \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae_get_caller6.c \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae_trace_callers.c \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae_user_function.c \
	   $(DESTDIR)$(datadir)/tests/overhead/papi_read1.c \
	   $(DESTDIR)$(datadir)/tests/overhead/papi_read4.c \
	   $(DESTDIR)$(datadir)/tests/overhead/run_overhead_tests.sh \
	   $(DESTDIR)$(datadir)/tests/overhead/extrae.xml \
	   $(DESTDIR)$(datadir)/tests/overhead/JavaEvent.java \
	   $(DESTDIR)$(datadir)/tests/overhead/JavaNEvent4.java \
	   $(DESTDIR)$(datadir)/tests/overhead/JavaFakeRoutine.java \
	   $(DESTDIR)$(datadir)/tests/overhead/extraej.xml \
	   $(DESTDIR)$(datadir)/tests/overhead/extraej_functionlist
	rmdir --ignore-fail-on-non-empty $(DESTDIR)$(datadir)/tests/overhead
