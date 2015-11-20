include $(top_srcdir)/PATHS

allTESTS = \
	check_Extrae_PAPI_TOT_INS.sh \
	check_Extrae_PAPI_TOT_CYC.sh \
	check_Extrae_PAPI_TOT_INS_CYC.sh

EXTRA_DIST = \
    $(allTESTS) \
    extrae-PAPI_TOT_CYC.xml \
    extrae-PAPI_TOT_INS.xml \
    extrae-PAPI_TOT_INS_CYC.xml

TESTS = \
    $(allTESTS)

check_PROGRAMS = \
	check_Extrae_counters_xml

check_Extrae_counters_xml_SOURCES = check_Extrae_counters_xml.c
check_Extrae_counters_xml_CFLAGS = -I$(INCLUDE_DIR) -I$(COMMON_DIR) -g
check_Extrae_counters_xml_LDFLAGS = -L$(TRACER_LIB) -lseqtrace

