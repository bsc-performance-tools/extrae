include $(top_srcdir)/PATHS

# Wrappers for openshmem instrumentation
WRAPPERS_OPENSHMEM = \
  openshmem_wrappers.c openshmem_wrappers.h \
  openshmem_probes.c openshmem_probes.h \
  openshmem_trace_macros.h

noinst_LTLIBRARIES  = libwrap_openshmem.la

libwrap_openshmem_la_CFLAGS = \
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
 @OPENSHMEM_CFLAGS@

libwrap_openshmem_la_SOURCES = $(WRAPPERS_OPENSHMEM)

