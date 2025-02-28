include $(top_srcdir)/PATHS

ALL_TESTS = \
 omp_target.sh

EXTRA_DIST = \
 $(ALL_TESTS) \
 trace.sh \
 extrae.xml

check_PROGRAMS = \
 omp_target

TESTS = $(ALL_TESTS)

omp_target_SOURCES = omp_target.c
omp_target_CFLAGS = -fopenmp -foffload=disable
