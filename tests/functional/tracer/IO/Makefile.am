include $(top_srcdir)/PATHS

ALL_TESTS = \
  io.sh     \
  fio.sh

EXTRA_DIST =   \
  $(ALL_TESTS) \
  trace.sh     \
  extrae.xml

check_PROGRAMS = \
  io             \
  fio

TESTS = $(ALL_TESTS)

TESTS_ENVIRONMENT = \
  EXTRAE_HOME=$(top_builddir)

io_SOURCES = io.c
fio_SOURCES = fio.c
