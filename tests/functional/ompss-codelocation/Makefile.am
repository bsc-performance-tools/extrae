include $(top_srcdir)/PATHS

check_PROGRAMS = ompss-codelocation ompss-codelocation-2

TESTS = ompss-codelocation.sh ompss-codelocation-2.sh

ompss_codelocation_SOURCES = ompss-codelocation.c
ompss_codelocation_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
ompss_codelocation_LDADD = -L$(TRACER_DIR) -lseqtrace

ompss_codelocation_2_SOURCES = ompss-codelocation-2.c
ompss_codelocation_2_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
ompss_codelocation_2_LDADD = -L$(TRACER_DIR) -lseqtrace

