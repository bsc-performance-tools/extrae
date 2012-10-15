include $(top_srcdir)/PATHS

check_PROGRAMS = define_event_type_gen_pcf

TESTS = define_event_type_gen_pcf.sh

define_event_type_gen_pcf_SOURCES = define_event_type_gen_pcf.c
define_event_type_gen_pcf_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
define_event_type_gen_pcf_LDADD = -L$(TRACER_DIR) -lseqtrace

