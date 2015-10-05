include $(top_srcdir)/PATHS

# Wrappers for fork instrumentation
WRAPPERS_FORK = \
 fork_wrapper.c fork_wrapper.h \
 fork_probe.c fork_probe.h 

noinst_LTLIBRARIES  = libwrap_fork.la

libwrap_fork_la_CFLAGS = \
 -I$(COMMON_MPI_DIR) \
 -I$(TRACER_INC) \
 -I$(HWC_INC) \
 -I$(CLOCKS_INC) \
 -I$(INTERFACES_INC)/API \
 -I$(WRAPPERS_INC)/API \
 -I$(top_srcdir) \
 -I$(top_srcdir)/include \
 -I$(SAMPLING_DIR)/timer \
 -I$(COMMON_INC) \
 $(UNWIND_CFLAGS)
libwrap_fork_la_SOURCES = $(WRAPPERS_FORK)

