include $(top_srcdir)/PATHS

# Wrappers for opencl instrumentation
WRAPPERS_OPENCL = \
 opencl_wrapper.c opencl_wrapper.h \
 opencl_common.c opencl_common.h \
 opencl_probe.c opencl_probe.h

noinst_LTLIBRARIES  = libwrap_opencl.la

libwrap_opencl_la_CFLAGS = \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC) \
 $(UNWIND_CFLAGS) \
-I$(OPENCL_HOME)/include

libwrap_opencl_la_SOURCES = $(WRAPPERS_OPENCL)

