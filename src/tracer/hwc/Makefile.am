include $(top_srcdir)/PATHS

### TEMPORARILY_DISABLED
#SUBDIRS = myrinet_hwc
#
#HWC = myrinet_hwc.c myrinet_hwc.h
HWC = hwc.h

EXTRA_DIST = hpm_rf.s

if HAVE_PAPI
HWC += common_hwc.c common_hwc.h \
 papi_hwc.c papi_hwc.h
else
if HAVE_PMAPI
HWC += common_hwc.c common_hwc.h \
 pmapi_hwc.c pmapi_hwc.h
else
HWC += fake_hwc.c
endif
endif

noinst_LTLIBRARIES = libhwc.la

libhwc_la_SOURCES = $(HWC)
libhwc_la_CFLAGS = -I$(HWC_INC) -I$(COMMON_INC) -I$(CLOCKS_INC) -I$(TRACER_INC) -I$(INCLUDE_DIR) -I$(top_srcdir)/src/tracer/wrappers/API @PAPI_CFLAGS@
