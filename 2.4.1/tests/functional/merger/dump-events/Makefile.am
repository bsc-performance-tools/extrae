include $(top_srcdir)/PATHS

check_PROGRAMS = dump-events

TESTS = dump-events.sh 

EXTRA_DIST = \
 dump-events.reference \
 dump-events.sh

if HAVE_PAPI
  PAPI_LINKER_FLAGS = -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
if HAVE_PMAPI
  PMAPI_LINKER_FLAGS = -lpmapi
endif
HWC_LINKER_FLAGS = $(PAPI_LINKER_FLAGS) $(PMAPI_LINKER_FLAGS)

dump_events_SOURCES = dump-events.c
dump_events_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
dump_events_LDFLAGS = -L$(TRACER_DIR) -lseqtrace $(HWC_LINKER_FLAGS)

