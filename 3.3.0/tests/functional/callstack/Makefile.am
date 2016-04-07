include $(top_srcdir)/PATHS

allTESTS = \
    check_Extrae_getcaller_depth2.sh \
	check_Extrae_user_function.sh

EXTRA_DIST = \
    $(allTESTS)

TESTS = \
    $(allTESTS)

check_PROGRAMS = \
	check_Extrae_getcaller_depth2 \
	check_Extrae_user_function

check_Extrae_getcaller_depth2_SOURCES = check_Extrae_getcaller_depth2.c
check_Extrae_getcaller_depth2_CFLAGS = -I$(INCLUDE_DIR) -g
check_Extrae_getcaller_depth2_LDFLAGS = -L$(TRACER_LIB) -lseqtrace

check_Extrae_user_function_SOURCES = check_Extrae_user_function.c
check_Extrae_user_function_CFLAGS = -I$(INCLUDE_DIR) -I$(COMMON_DIR) -g
check_Extrae_user_function_LDFLAGS = -L$(TRACER_LIB) -lseqtrace

