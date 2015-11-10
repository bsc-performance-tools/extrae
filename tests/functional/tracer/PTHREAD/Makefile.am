include $(top_srcdir)/PATHS

EXTRA_DIST= \
 pthread.sh \
 trace-ldpreload.sh \
 extrae.xml

check_PROGRAMS = \
 pthread

TESTS = \
  pthread.sh

pthread_SOURCES = pthread.c
pthread_CFLAGS = -g -pthread
