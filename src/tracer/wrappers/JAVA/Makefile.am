include $(top_srcdir)/PATHS

# Wrappers for Java instrumentation
WRAPPERS_JAVA_ = \
 java_probe.c java_probe.h 

noinst_LTLIBRARIES  = libwrap_java.la

libwrap_java_la_CFLAGS = \
 -I$(COMMON_DIR) \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_DIR) \
 -I$(HWC_DIR) \
 -I$(CLOCKS_DIR) \
 -I$(INTERFACES_DIR)/API \
 -I$(WRAPPERS_DIR)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_DIR) \
 $(UNWIND_CFLAGS)
libwrap_java_la_SOURCES = $(WRAPPERS_JAVA_)

