include $(top_srcdir)/PATHS

check_PROGRAMS = main
main_SOURCES = main.c

TESTS = test-value-range.sh

TESTS_ENVIRONMENT = \
  EXTRAE_HOME=$(top_builddir)

EXTRA_DIST = \
  test-value-range.sh

main$(EXEEXT): main.c
	gcc -g $^ -Wl,-rpath -Wl,../../../../src/tracer/.libs -Wl,-rpath -Wl,. -L ../../../../src/tracer/.libs -lseqtrace -o $@

clean-local:
	-rm -rf set-0/ TRACE.mpits main.{pcf,prv,row}
	-rm -rf lib*