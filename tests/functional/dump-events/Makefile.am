include $(top_srcdir)/PATHS

check_PROGRAMS = check-dump-events

TESTS = check-dump-events.sh

check_dump_events_SOURCES = check-dump-events.c
check_dump_events_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
check_dump_events_LDADD = -L$(TRACER_DIR) -lseqtrace

