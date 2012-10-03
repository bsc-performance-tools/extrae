include $(top_srcdir)/PATHS

check_PROGRAMS = check-ompss-codelocation

TESTS = check-ompss-codelocation-1.sh check-ompss-codelocation-2.sh

check_ompss_codelocation_SOURCES = check-ompss-codelocation.c
check_ompss_codelocation_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
check_ompss_codelocation_LDADD = -L$(TRACER_DIR) -lseqtrace

