include $(top_srcdir)/PATHS

check_PROGRAMS = \
 auto-init-fini \
 define_event_type_gen_pcf \
 define_event_type_gen_pcf_f
# ompss-codelocation \
# ompss-codelocation-2

TESTS = \
 auto-init-fini.sh \
 define_event_type_gen_pcf.sh \
 define_event_type_gen_pcf_f.sh
# ompss-codelocation.sh \
# ompss-codelocation-2.sh

EXTRA_DIST = \
 auto-init-fini.sh \
 define_event_type_gen_pcf.sh \
 define_event_type_gen_pcf_f.sh \
 ompss-codelocation-2.sh \
 ompss-codelocation.sh

if HAVE_PAPI
if !HAVE_PAPI_EMBEDDED
  PAPI_LINKER_FLAGS = -L@PAPI_SHAREDLIBSDIR@ -R @PAPI_SHAREDLIBSDIR@ @PAPI_LIBS@
endif
endif
if HAVE_PMAPI
  PMAPI_LINKER_FLAGS = -lpmapi
endif
HWC_LINKER_FLAGS = $(PAPI_LINKER_FLAGS) $(PMAPI_LINKER_FLAGS)

auto_init_fini_SOURCES = auto-init-fini.c
auto_init_fini_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
auto_init_fini_LDFLAGS = -L$(TRACER_LIB) -lseqtrace $(HWC_LINKER_FLAGS)

define_event_type_gen_pcf_SOURCES = define_event_type_gen_pcf.c
define_event_type_gen_pcf_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
define_event_type_gen_pcf_LDFLAGS = -L$(TRACER_LIB) -lseqtrace $(HWC_LINKER_FLAGS)

define_event_type_gen_pcf_f_SOURCES = $(INCLUDE_DIR)/extrae_module.f90 define_event_type_gen_pcf.f90
define_event_type_gen_pcf_f_FCFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
define_event_type_gen_pcf_f_LDFLAGS = -L$(TRACER_LIB) -lseqtrace $(HWC_LINKER_FLAGS)

# These 2 are disabled because they produce different results in different machines
#
#ompss_codelocation_SOURCES = ompss-codelocation.c
#ompss_codelocation_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
#ompss_codelocation_LDFLAGS = -L$(TRACER_LIB) -lseqtrace $(HWC_LINKER_FLAGS)
#
#ompss_codelocation_2_SOURCES = ompss-codelocation-2.c
#ompss_codelocation_2_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
#ompss_codelocation_2_LDFLAGS = -L$(TRACER_LIB) -lseqtrace $(HWC_LINKER_FLAGS)
#
