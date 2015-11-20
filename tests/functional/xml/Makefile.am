include $(top_srcdir)/PATHS

ALLTESTS = \
	check_Extrae_xml.sh \
	check_Extrae_xml_envvar_merge.sh \
	check_Extrae_xml_envvar_counters.sh

EXTRA_DIST = \
    $(ALLTESTS) \
	extrae_envvar_counters.xml \
	extrae_envvar_merge.xml

TESTS = \
    $(ALLTESTS)

check_PROGRAMS = \
	check_Extrae_xml

check_Extrae_xml_SOURCES = check_Extrae_xml.c
check_Extrae_xml_CFLAGS = -I$(INCLUDE_DIR) -I$(COMMON_DIR) -g
check_Extrae_xml_LDFLAGS = -L$(TRACER_LIB) -lseqtrace

