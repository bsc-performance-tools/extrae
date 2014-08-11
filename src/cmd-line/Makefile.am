include ../../PATHS

bin_PROGRAMS = extrae-cmd

extrae_cmd_SOURCES = \
 extrae-cmd.c extrae-cmd.h \
 extrae-cmd-init.c extrae-cmd-init.h \
 extrae-cmd-emit.c extrae-cmd-emit.h \
 extrae-cmd-fini.c extrae-cmd-fini.h

extrae_cmd_CFLAGS = -I$(INCLUDE_DIR) -I$(COMMON_DIR) -I$(TRACER_DIR) \
 -I$(TRACER_DIR)/wrappers/API -I$(TRACER_DIR)/clocks -I$(TRACER_DIR)/hwc -O -g

extrae_cmd_LDADD = $(TRACER_DIR)/libseqtrace.la $(MERGER_DIR)/libmpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS) -lm

if IS_BG_MACHINE
extrae_cmd_LDADD += $(XML2_LDFLAGS) $(XML2_LIBS)
extrae_cmd_LDFLAGS = -all-static
endif
