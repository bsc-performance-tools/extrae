include $(top_srcdir)/PATHS

check_PROGRAMS = auto-init-fini

TESTS = auto-init-fini.sh

auto_init_fini_SOURCES = auto-init-fini.c
auto_init_fini_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
auto_init_fini_LDADD = -L$(TRACER_DIR) -lseqtrace

