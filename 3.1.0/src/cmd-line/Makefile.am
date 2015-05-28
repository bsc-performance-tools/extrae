include ../../PATHS

bin_PROGRAMS = extrae-cmd

extrae_cmd_SOURCES = \
 extrae-cmd.c extrae-cmd.h \
 extrae-cmd-init.c extrae-cmd-init.h \
 extrae-cmd-emit.c extrae-cmd-emit.h \
 extrae-cmd-fini.c extrae-cmd-fini.h

extrae_cmd_CFLAGS = -I$(INCLUDE_DIR) -I$(COMMON_INC) -I$(TRACER_INC) \
 -I$(TRACER_INC)/wrappers/API -I$(TRACER_INC)/clocks -I$(TRACER_INC)/hwc -O -g

extrae_cmd_LDADD = $(TRACER_LIB)/libseqtrace.la $(MERGER_LIB)/libmpi2prv.la $(BFD_LDFLAGS) $(BFD_LIBS) $(LIBERTY_LDFLAGS) $(LIBERTY_LIBS) -lm

extrae_cmd_LDFLAGS = @PAPI_LDFLAGS@ @PAPI_LIBS@
if IS_BG_MACHINE
if HAVE_PAPI
extrae_cmd_LDFLAGS += -all-static -L@PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
extrae_cmd_LDADD += $(XML2_LDFLAGS) $(XML2_LIBS)
else
if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
extrae_cmd_LDFLAGS += -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
endif
endif
