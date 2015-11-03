include $(top_srcdir)/PATHS

# Wrappers for IO instrumentation
WRAPPERS_IO = \
 io_wrapper.c io_wrapper.h \
 io_probe.c io_probe.h 

noinst_LTLIBRARIES  = libwrap_io.la

libwrap_io_la_CFLAGS = \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(COMMON_INC) \
 $(UNWIND_CFLAGS)
libwrap_io_la_SOURCES = $(WRAPPERS_IO)

