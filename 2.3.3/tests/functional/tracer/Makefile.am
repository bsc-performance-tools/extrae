include $(top_srcdir)/PATHS

check_PROGRAMS = \
 auto-init-fini \
 define_event_type_gen_pcf \
 ompss-codelocation \
 ompss-codelocation-2

TESTS = \
 auto-init-fini.sh \
 define_event_type_gen_pcf.sh \
 ompss-codelocation.sh \
 ompss-codelocation-2.sh

EXTRA_DIST = \
 auto-init-fini.reference \
 auto-init-fini.sh \
 define_event_type_gen_pcf.reference \
 define_event_type_gen_pcf.sh \
 ompss-codelocation-2.reference \
 ompss-codelocation-2.sh \
 ompss-codelocation.reference \
 ompss-codelocation.sh

auto_init_fini_SOURCES = auto-init-fini.c
auto_init_fini_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
auto_init_fini_LDADD = -L$(TRACER_DIR) -lseqtrace

define_event_type_gen_pcf_SOURCES = define_event_type_gen_pcf.c
define_event_type_gen_pcf_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
define_event_type_gen_pcf_LDADD = -L$(TRACER_DIR) -lseqtrace

ompss_codelocation_SOURCES = ompss-codelocation.c
ompss_codelocation_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
ompss_codelocation_LDADD = -L$(TRACER_DIR) -lseqtrace

ompss_codelocation_2_SOURCES = ompss-codelocation-2.c
ompss_codelocation_2_CFLAGS = -I$(INCLUDE_DIR) -I$(SRC_DIR)/common
ompss_codelocation_2_LDADD = -L$(TRACER_DIR) -lseqtrace

