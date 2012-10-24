include $(top_srcdir)/PATHS

check_PROGRAMS = dump-events

TESTS = dump-events.sh

dump_events_SOURCES = dump-events.c
dump_events_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
dump_events_LDADD = -L$(TRACER_DIR) -lseqtrace

