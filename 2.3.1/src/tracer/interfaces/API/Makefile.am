include $(top_srcdir)/PATHS

iface_API_SRCS     = misc_interface.c misc_interface.h
iface_API_INCLUDES = \
   -I$(COMMON_DIR)   \
   -I$(CLOCKS_DIR)   \
   -I$(TRACER_DIR)   \
   -I$(HWC_DIR)      \
   -I$(top_srcdir)/include \
   -I$(WRAPPERS_DIR)/API @PAPI_CFLAGS@

noinst_LTLIBRARIES      = libiface_api.la libiface_api_pthread.la

libiface_api_la_SOURCES = $(iface_API_SRCS)
libiface_api_la_CFLAGS  = $(iface_API_INCLUDES)

if WANT_PTHREAD
libiface_api_pthread_la_SOURCES = $(iface_API_SRCS)
libiface_api_pthread_la_CFLAGS  = $(iface_API_INCLUDES) -DPTHREAD_SUPPORT
endif
