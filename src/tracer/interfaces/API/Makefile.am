include $(top_srcdir)/PATHS

iface_API_SRCS     = misc_interface.c misc_interface.h
iface_API_INCLUDES = \
   -I$(COMMON_INC)   \
   -I$(CLOCKS_INC)   \
   -I$(TRACER_INC)   \
   -I$(HWC_INC)      \
   -I$(top_srcdir)/include \
   -I$(WRAPPERS_INC)/API @PAPI_CFLAGS@

noinst_LTLIBRARIES = libiface_api.la
if WANT_PTHREAD
  noinst_LTLIBRARIES += libiface_api_pthread.la
endif

libiface_api_la_SOURCES = $(iface_API_SRCS)
libiface_api_la_CFLAGS  = $(iface_API_INCLUDES)

if WANT_PTHREAD
  libiface_api_pthread_la_SOURCES = $(iface_API_SRCS)
  libiface_api_pthread_la_CFLAGS  = $(iface_API_INCLUDES) -DPTHREAD_SUPPORT
endif
