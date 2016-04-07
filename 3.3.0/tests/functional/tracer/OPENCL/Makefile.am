include $(top_srcdir)/PATHS

EXTRA_DIST = \
 opencl-check.pcf.reference \
 opencl-check.sh \
 extrae.xml

check_PROGRAMS = \
 opencl-check

TESTS = \
 opencl-check.sh

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  PAPI_LINKER_FLAGS = -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
endif
if HAVE_PMAPI
  PMAPI_LINKER_FLAGS = -lpmapi
endif
HWC_LINKER_FLAGS = $(PAPI_LINKER_FLAGS) $(PMAPI_LINKER_FLAGS)

opencl_check_SOURCES = opencl-check.c
opencl_check_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common @OPENCL_CFLAGS@
opencl_check_LDFLAGS = -L$(TRACER_LIB) -locltrace $(HWC_LINKER_FLAGS) -L@OPENCL_SHAREDLIBSDIR@ -R @OPENCL_SHAREDLIBSDIR@ -lOpenCL
