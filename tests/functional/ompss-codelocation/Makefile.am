include $(top_srcdir)/PATHS

check_PROGRAMS = ompss_codelocation

TESTS = ompss-codelocation.sh

ompss_codelocation_SOURCES = ompss-codelocation.c
ompss_codelocation_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
ompss_codelocation_LDADD = -L$(TRACER_DIR) -lseqtrace

