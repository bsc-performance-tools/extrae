include $(top_srcdir)/PATHS

# Wrappers for pthread instrumentation
WRAPPERS_PTHREAD = \
 pthread_wrapper.c pthread_wrapper.h \
 pthread_probe.c pthread_probe.h 

noinst_LTLIBRARIES  = libwrap_pthread.la

libwrap_pthread_la_CFLAGS = \
 -DPTHREAD_SUPPORT \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC) \
 $(UNWIND_CFLAGS) $(PTHREAD_CFLAGS)
libwrap_pthread_la_SOURCES = $(WRAPPERS_PTHREAD)

