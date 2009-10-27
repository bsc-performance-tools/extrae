include $(top_srcdir)/PATHS

SUBDIRS = myrinet_hwc

HWC = myrinet_hwc.c

if HAVE_PAPI
HWC += common_hwc.c papi_hwc.c
else
if HAVE_PMAPI
HWC += common_hwc.c pmapi_hwc.c
else
HWC += fake_hwc.c
endif
endif

noinst_LTLIBRARIES = libhwc.la

libhwc_la_SOURCES = $(HWC)
libhwc_la_CFLAGS = -I. -I$(COMMON_DIR) -I$(CLOCKS_DIR) -I$(TRACER_DIR) -I$(top_srcdir)/src/tracer/wrappers/API @PAPI_CFLAGS@
