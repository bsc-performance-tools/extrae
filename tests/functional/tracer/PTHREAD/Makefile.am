include $(top_srcdir)/PATHS

EXTRA_DIST= \
 pthread.sh \
 trace-ldpreload.sh \
 extrae.xml

check_PROGRAMS =
TESTS =

if HAVE_LIBADDR2LINE
check_PROGRAMS += \
 pthread

TESTS += \
  pthread.sh
endif

TESTS_ENVIRONMENT = \
  EXTRAE_HOME=$(top_builddir)

pthread_SOURCES = pthread.c
pthread_CFLAGS = -g -pthread
