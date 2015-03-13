SUBDIRS = jni

include $(top_srcdir)/PATHS

lib_LTLIBRARIES = libjavaseqtrace.la

libjavaseqtrace_la_SOURCES = jni/extrae_wrapper.c
libjavaseqtrace_la_CFLAGS  = $(JAVA_INCLUDES) -Ijni/include -I$(INCLUDE_DIR) -I$(COMMON_INC) -I$(BASE_DIR) $(CFLAGS)
libjavaseqtrace_la_LIBADD  = $(TRACER_LIB)/libseqtrace.la

