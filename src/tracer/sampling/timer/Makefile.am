include $(top_srcdir)/PATHS

sampling_timer_API_SRCS = \
 sampling-timer.c sampling-timer.h

iface_API_INCLUDES = \
   -I$(COMMON_INC)   \
   -I$(CLOCKS_INC)   \
   -I$(TRACER_INC)   \
   -I$(HWC_INC)      \
   -I$(top_srcdir)/include \
   -I$(SAMPLING_DIR)/common \
   -I$(WRAPPERS_INC)/API @PAPI_CFLAGS@

noinst_LTLIBRARIES = libsampling_timer.la

libsampling_timer_la_SOURCES = $(sampling_timer_API_SRCS)
libsampling_timer_la_CFLAGS  = $(iface_API_INCLUDES)

