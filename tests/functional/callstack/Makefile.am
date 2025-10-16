include $(top_srcdir)/PATHS

allTESTS = \
    check_Extrae_getcaller_depth2.sh \
	check_Extrae_user_function.sh

EXTRA_DIST = \
    $(allTESTS)

TESTS = \
    $(allTESTS)

TESTS_ENVIRONMENT = \
  EXTRAE_HOME=$(top_builddir)

check_PROGRAMS = \
	check_Extrae_getcaller_depth2 \
	check_Extrae_user_function

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  PAPI_LINKER_FLAGS = @PAPI_LDFLAGS@ @PAPI_RPATH@ @PAPI_LIBS@
endif
endif
if HAVE_PMAPI
  PMAPI_LINKER_FLAGS = -lpmapi
endif
HWC_LINKER_FLAGS = $(PAPI_LINKER_FLAGS) $(PMAPI_LINKER_FLAGS)

check_Extrae_getcaller_depth2_SOURCES = check_Extrae_getcaller_depth2.c
check_Extrae_getcaller_depth2_CFLAGS = -I$(INCLUDE_DIR) -g
check_Extrae_getcaller_depth2_LDFLAGS = -L$(TRACER_LIB) -lseqtrace $(HWC_LINKER_FLAGS)

check_Extrae_user_function_SOURCES = check_Extrae_user_function.c
check_Extrae_user_function_CFLAGS = -I$(INCLUDE_DIR) -I$(COMMON_DIR) -g
check_Extrae_user_function_LDFLAGS = -L$(TRACER_LIB) -lseqtrace $(HWC_LINKER_FLAGS)

