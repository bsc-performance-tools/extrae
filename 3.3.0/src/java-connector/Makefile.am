SUBDIRS = jni

if WANT_JAVA_JVMTI
SUBDIRS += jvmti-agent
endif

include $(top_srcdir)/PATHS

lib_LTLIBRARIES = \
  libjavatrace.la

libjavatrace_la_SOURCES = jni/extrae_wrapper.c
libjavatrace_la_CFLAGS  = $(JAVA_INCLUDES)\
 -Ijni/include \
 -I$(INCLUDE_DIR) \
 -I$(COMMON_INC) \
 -I$(BASE_DIR) \
 -I$(TRACER_INC) \
 -I$(WRAPPERS_INC)/API \
 -I$(CLOCKS_INC) \
 -I$(HWC_INC) \
 $(CFLAGS)
libjavatrace_la_LIBADD  = $(TRACER_LIB)/libpttrace.la
libjavatrace_la_LDFLAGS = -release $(PACKAGE_VERSION)
