
include $(top_srcdir)/PATHS

sampling_common_API_SRCS = \
 sampling-common.c sampling-common.h

iface_API_INCLUDES = \
   -I$(COMMON_INC)   \
   -I$(CLOCKS_INC)   \
   -I$(TRACER_INC)   \
   -I$(HWC_INC)      \
   -I$(top_srcdir)/include \
   -I$(WRAPPERS_INC)/API @PAPI_CFLAGS@

noinst_LTLIBRARIES = libsampling_common.la

libsampling_common_la_SOURCES = $(sampling_common_API_SRCS)
libsampling_common_la_CFLAGS  = $(iface_API_INCLUDES)

