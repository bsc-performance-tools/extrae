include $(top_srcdir)/PATHS

check_PROGRAMS = check-auto-init-fini

TESTS = check-auto-init-fini.sh

check_auto_init_fini_SOURCES = check-auto-init-fini.c
check_auto_init_fini_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
check_auto_init_fini_LDADD = -L$(TRACER_DIR) -lseqtrace

