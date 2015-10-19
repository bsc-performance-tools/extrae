include $(top_srcdir)/PATHS

sampling_intel_pebs_API_SRCS = \
 sampling-intel-pebs.c sampling-intel-pebs.h \
 perf_event.h

iface_API_INCLUDES = \
   -I$(COMMON_INC)   \
   -I$(CLOCKS_INC)   \
   -I$(TRACER_INC)   \
   -I$(HWC_INC)      \
   -I$(top_srcdir)/include \
   -I$(SAMPLING_DIR)/common \
   -I$(WRAPPERS_INC)/API @PAPI_CFLAGS@

noinst_LTLIBRARIES = libsampling_intel_pebs.la

libsampling_intel_pebs_la_SOURCES = $(sampling_intel_pebs_API_SRCS)
libsampling_intel_pebs_la_CFLAGS  = $(iface_API_INCLUDES)

