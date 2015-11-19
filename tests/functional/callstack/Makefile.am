include $(top_srcdir)/PATHS

check_PROGRAMS = \
	check_Extrae_getcaller_depth2

TESTS = \
    check_Extrae_getcaller_depth2.sh

check_Extrae_getcaller_depth2_SOURCES = check_Extrae_getcaller_depth2.c
check_Extrae_getcaller_depth2_CFLAGS = -I$(INCLUDE_DIR) -g
check_Extrae_getcaller_depth2_LDFLAGS = -L$(TRACER_LIB) -lseqtrace

